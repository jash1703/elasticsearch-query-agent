import os
import logging
from typing import List, Union, Literal, Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator, field_validator
from elasticsearch import AsyncElasticsearch

load_dotenv()
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST")
USERNAME = os.getenv("ELASTIC_USERNAME")
PASSWORD = os.getenv("PASSWORD")
INDEX_PATTERN = os.getenv("INDEX_PATTERN")

ALLOWED_FIELDS = os.getenv("ALLOWED_FIELDS")

ALLOWED_OPERATORS = {"eq", "ne", "range", "geo_distance"}
LOGICAL_OPS = {"AND", "OR"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("es_query_api")


Operator = Literal["eq", "ne", "range", "geo_distance"]
LogicalOp = Literal["AND", "OR"]

class RangeValue(BaseModel):
    gte: str
    lte: str

class GeoValue(BaseModel):
    lat: float
    lon: float
    distance: str

class Condition(BaseModel):
    field: str
    operator: Operator
    value: Union[int, float, str, RangeValue, GeoValue]

    @field_validator("value", mode="before")
    def coerce_complex(cls, v, info):
        op = info.data.get("operator")
        if op == "range" and isinstance(v, dict):
            return RangeValue(**v)
        if op == "geo_distance" and isinstance(v, dict):
            return GeoValue(**v)
        return v

    @model_validator(mode="after")
    def check_field_and_value(self):
        if self.field not in ALLOWED_FIELDS:
            raise ValueError(f"Field '{self.field}' is not queryable")
        if self.operator not in ALLOWED_OPERATORS:
            raise ValueError(f"Operator '{self.operator}' not supported")
        if self.operator == "range" and not isinstance(self.value, RangeValue):
            raise ValueError("range requires a RangeValue")
        if self.operator == "geo_distance" and not isinstance(self.value, GeoValue):
            raise ValueError("geo_distance requires a GeoValue")
        return self

class FilterGroup(BaseModel):
    filters: List[Union[Condition, "FilterGroup"]]
    logical_operator: LogicalOp = Field("AND", description="Combine sub-filters with AND or OR")

    @model_validator(mode="after")
    def check_logical_operator(self):
        if self.logical_operator not in LOGICAL_OPS:
            raise ValueError(f"Logical operator '{self.logical_operator}' not supported")
        return self

FilterGroup.update_forward_refs()

AggType = Literal[
    # Bucket aggregations
    "terms", "date_histogram", "histogram", "range", "filters", "composite",
    # Metric aggregations  
    "avg", "sum", "min", "max", "cardinality", "value_count", "stats", "extended_stats",
    # Pipeline aggregations
    "bucket_selector", "bucket_sort", "avg_bucket", "sum_bucket", "min_bucket", 
    "max_bucket", "stats_bucket", "extended_stats_bucket"
]

class TermsInclude(BaseModel):
    partition: int
    num_partitions: int

class RangeSpec(BaseModel):
    key: Optional[str] = None
    from_: Optional[float] = Field(None, alias="from")
    to: Optional[float] = None

class FilterSpec(BaseModel):
    name: str
    filter: Dict[str, Any]

class CompositeSource(BaseModel):
    name: str
    terms: Optional[Dict[str, Any]] = None
    date_histogram: Optional[Dict[str, Any]] = None
    histogram: Optional[Dict[str, Any]] = None

class Aggregation(BaseModel):
    name: str
    type: AggType
    field: Optional[str] = None
    missing: Optional[Union[str, int, float]] = None
    size: Optional[int] = None
    order: Optional[Dict[str, Literal["asc", "desc"]]] = None
    include: Optional[Union[str, TermsInclude]] = None
    exclude: Optional[str] = None
    min_doc_count: Optional[int] = None
    shard_min_doc_count: Optional[int] = None
    calendar_interval: Optional[str] = None
    fixed_interval: Optional[str] = None
    format: Optional[str] = None
    time_zone: Optional[str] = None
    offset: Optional[str] = None
    interval: Optional[float] = None
    ranges: Optional[List[RangeSpec]] = None
    keyed: Optional[bool] = None
    filters_spec: Optional[List[FilterSpec]] = Field(None, alias="filters")
    other_bucket: Optional[bool] = None
    other_bucket_key: Optional[str] = None
    sources: Optional[List[CompositeSource]] = None
    after: Optional[Dict[str, Any]] = None
    buckets_path: Optional[Union[str, Dict[str, str]]] = None
    script: Optional[Union[str, Dict[str, Any]]] = None
    gap_policy: Optional[Literal["skip", "insert_zeros"]] = None
    script_params: Optional[Dict[str, Any]] = None
    sort: Optional[List[Dict[str, Any]]] = None
    from_: Optional[int] = Field(None, alias="from")
    sub_aggregations: List["Aggregation"] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_aggregation(self):
        field_required_types = {
            "terms", "date_histogram", "histogram", "range",
            "avg", "sum", "min", "max", "cardinality", "value_count", "stats", "extended_stats"
        }
        pipeline_types = {
            "bucket_selector", "bucket_sort", "avg_bucket", "sum_bucket", 
            "min_bucket", "max_bucket", "stats_bucket", "extended_stats_bucket"
        }
        if self.type == "terms" and not (self.field or self.script):
            raise ValueError("Aggregation 'terms' requires a 'field' or a 'script'")
        elif self.type in field_required_types and not self.field and self.type != "terms":
            raise ValueError(f"Aggregation '{self.type}' requires a 'field'")
        if self.type in pipeline_types and not self.buckets_path:
            raise ValueError(f"Pipeline aggregation '{self.type}' requires 'buckets_path'")
        if self.type == "terms" and self.size is None:
            self.size = 10
        if self.type == "date_histogram":
            if not self.calendar_interval and not self.fixed_interval:
                raise ValueError("date_histogram requires either 'calendar_interval' or 'fixed_interval'")
        if self.type == "histogram" and self.interval is None:
            raise ValueError("histogram requires 'interval'")
        if self.type == "range" and not self.ranges:
            raise ValueError("range aggregation requires 'ranges'")
        if self.type == "filters" and not self.filters_spec:
            raise ValueError("filters aggregation requires 'filters'")
        if self.type == "composite" and not self.sources:
            raise ValueError("composite aggregation requires 'sources'")
        if self.type == "bucket_selector" and not self.script:
            raise ValueError("bucket_selector requires 'script'")
        return self

Aggregation.model_rebuild()


class QueryRequest(BaseModel):
    filters: List[Union[Condition, FilterGroup]]
    logical_operator: LogicalOp = Field("AND", description="Combine conditions with AND or OR")
    size: int = Field(1000, description="Batch size per scroll request, 0 for aggregations only")
    scroll: str = Field("2m", description="Scroll context lifetime")
    indices: str = Field(INDEX_PATTERN, description="Index pattern to query")
    aggregations: Optional[List[Aggregation]] = Field(default=None, description="Dynamic aggregations")
    fields: Optional[List[str]] = Field(default=None, description="_source fields to return")
    composite_after: Optional[Dict[str, Any]] = Field(default=None, description="After key for composite pagination")

    @model_validator(mode="after")
    def validate_logical_op(self):
        if self.logical_operator not in LOGICAL_OPS:
            raise ValueError(f"Logical operator '{self.logical_operator}' not supported")
        return self

    @model_validator(mode="after") 
    def validate_fields(self):
        if self.fields:
            for f in self.fields:
                if f not in ALLOWED_FIELDS:
                    raise ValueError(f"Field '{f}' not in allowed _source fields")
        return self


def build_clause(item: Union[Condition, FilterGroup]) -> Dict[str, Any]:
    if isinstance(item, Condition):
        if item.operator == "eq":
            return {"term": {item.field: item.value}}
        if item.operator == "ne":
            return {"bool": {"must_not": {"term": {item.field: item.value}}}}
        if item.operator == "range" and isinstance(item.value, RangeValue):
            return {"range": {item.field: item.value.dict()}}
        if item.operator == "geo_distance" and isinstance(item.value, GeoValue):
            return {"geo_distance": {"distance": item.value.distance, item.field: {"lat": item.value.lat, "lon": item.value.lon}}}
        raise ValueError(f"Invalid Condition operator/value: {item.operator}")
    sub = [build_clause(s) for s in item.filters]
    key = "must" if item.logical_operator == "AND" else "should"
    bool_body = {key: sub}
    if item.logical_operator == "OR":
        bool_body["minimum_should_match"] = 1
    return {"bool": bool_body}

def build_single_agg(agg: Aggregation) -> Dict[str, Any]:
    if agg.type == "terms":
        terms_body = {"field": agg.field}
        if isinstance(agg.size, int):
            terms_body["size"] = agg.size
        if isinstance(agg.order, dict) and all(isinstance(k, str) and v in ("asc", "desc") for k, v in agg.order.items()):
            terms_body["order"] = agg.order
        if isinstance(agg.include, str):
            terms_body["include"] = agg.include
        if isinstance(agg.exclude, str):
            terms_body["exclude"] = agg.exclude
        if isinstance(agg.min_doc_count, int):
            terms_body["min_doc_count"] = agg.min_doc_count
        if isinstance(agg.include, TermsInclude):
            terms_body["include"] = {
                "partition": agg.include.partition,
                "num_partitions": agg.include.num_partitions
            }
        body = {"terms": terms_body}
    elif agg.type == "date_histogram":
        date_hist_body = {"field": agg.field}
        if agg.calendar_interval:
            date_hist_body["calendar_interval"] = agg.calendar_interval
        if agg.fixed_interval:
            date_hist_body["fixed_interval"] = agg.fixed_interval
        if agg.format:
            date_hist_body["format"] = agg.format
        if agg.time_zone:
            date_hist_body["time_zone"] = agg.time_zone
        if agg.offset:
            date_hist_body["offset"] = agg.offset
        if isinstance(agg.min_doc_count, int):
            date_hist_body["min_doc_count"] = agg.min_doc_count
        body = {"date_histogram": date_hist_body}
    elif agg.type == "histogram":
        hist_body = {"field": agg.field, "interval": agg.interval}
        if isinstance(agg.min_doc_count, int):
            hist_body["min_doc_count"] = agg.min_doc_count
        body = {"histogram": hist_body}
    elif agg.type == "range":
        range_body = {"field": agg.field, "ranges": []}
        for r in agg.ranges or []:
            range_item = {}
            if r.key:
                range_item["key"] = r.key
            if r.from_ is not None:
                range_item["from"] = r.from_
            if r.to is not None:
                range_item["to"] = r.to
            range_body["ranges"].append(range_item)
        if isinstance(agg.keyed, bool):
            range_body["keyed"] = agg.keyed
        body = {"range": range_body}
    elif agg.type == "filters":
        filters_body = {"filters": {}}
        for f in agg.filters_spec or []:
            filters_body["filters"][f.name] = f.filter
        if isinstance(agg.other_bucket, bool):
            filters_body["other_bucket"] = agg.other_bucket
        if isinstance(agg.other_bucket_key, str):
            filters_body["other_bucket_key"] = agg.other_bucket_key
        body = {"filters": filters_body}
    elif agg.type == "composite":
        composite_body = {"sources": []}
        for source in agg.sources or []:
            source_dict = {source.name: {}}
            if source.terms:
                source_dict[source.name]["terms"] = source.terms
            elif source.date_histogram:
                source_dict[source.name]["date_histogram"] = source.date_histogram
            elif source.histogram:
                source_dict[source.name]["histogram"] = source.histogram
            composite_body["sources"].append(source_dict)
        if isinstance(agg.size, int):
            composite_body["size"] = agg.size
        if isinstance(agg.after, dict):
            composite_body["after"] = agg.after
        body = {"composite": composite_body}
    elif agg.type in ["avg", "sum", "min", "max", "cardinality", "value_count"]:
        body = {agg.type: {"field": agg.field}}
    elif agg.type in ["stats", "extended_stats"]:
        body = {agg.type: {"field": agg.field}}
    elif agg.type == "bucket_selector":
        bucket_selector_body = {}
        if agg.buckets_path:
            bucket_selector_body["buckets_path"] = agg.buckets_path
        if agg.script:
            if isinstance(agg.script, str):
                bucket_selector_body["script"] = {"source": agg.script}
            else:
                bucket_selector_body["script"] = agg.script
        if agg.gap_policy:
            bucket_selector_body["gap_policy"] = agg.gap_policy
        body = {"bucket_selector": bucket_selector_body}
    elif agg.type == "bucket_sort":
        bucket_sort_body = {}
        if agg.sort:
            bucket_sort_body["sort"] = agg.sort
        if isinstance(agg.size, int):
            bucket_sort_body["size"] = agg.size
        if isinstance(agg.from_, int):
            bucket_sort_body["from"] = agg.from_
        body = {"bucket_sort": bucket_sort_body}
    elif agg.type in ["avg_bucket", "sum_bucket", "min_bucket", "max_bucket"]:
        metric_body = {"buckets_path": agg.buckets_path}
        if agg.gap_policy:
            metric_body["gap_policy"] = agg.gap_policy
        body = {agg.type: metric_body}
    elif agg.type in ["stats_bucket", "extended_stats_bucket"]:
        metric_body = {"buckets_path": agg.buckets_path}
        if agg.gap_policy:
            metric_body["gap_policy"] = agg.gap_policy
        body = {agg.type: metric_body}
    else:
        raise ValueError(f"Unsupported aggregation type: {agg.type}")
    if agg.missing is not None and agg.type not in ["bucket_selector", "bucket_sort"]:
        for agg_type_key in body.keys():
            if isinstance(body[agg_type_key], dict):
                body[agg_type_key]["missing"] = agg.missing
    return body

def build_aggs(aggs: List[Aggregation]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for agg in aggs:
        agg_body = build_single_agg(agg)
        if agg.sub_aggregations:
            agg_body["aggs"] = build_aggs(agg.sub_aggregations)
        result[agg.name] = agg_body
    return result

def build_es_query(req: QueryRequest) -> Dict[str, Any]:
    root = FilterGroup(filters=req.filters, logical_operator=req.logical_operator)
    body: Dict[str, Any] = {
        "query": build_clause(root),
        "sort": [{"timestamp": {"order": "desc"}}],
        "size": req.size
    }
    if req.fields:
        body["_source"] = req.fields
    if req.aggregations:
        body["aggs"] = build_aggs(req.aggregations)
    return body

app = FastAPI(title="Dynamic ES Query API with Aggs")

@app.on_event("startup")
async def startup():
    hosts = [ELASTICSEARCH_HOST] if ELASTICSEARCH_HOST else []
    basic_auth = (USERNAME, PASSWORD) if USERNAME and PASSWORD else None
    app.state.es = AsyncElasticsearch(
        hosts=hosts,
        basic_auth=basic_auth,
        verify_certs=False
    )

@app.on_event("shutdown")
async def shutdown():
    await app.state.es.close()

@app.post("/query")
async def query_es(req: QueryRequest):
    logger.info("Received request: %s", req.json())
    try:
        es_body = build_es_query(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    logger.info("Generated ES DSL: %s", es_body)
    es = app.state.es
    do_scroll = req.size > 0
    try:
        if do_scroll:
            resp = await es.search(index=req.indices, body=es_body, scroll=req.scroll)
        else:
            resp = await es.search(index=req.indices, body=es_body)
    except Exception as e:
        logger.error("ES search failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Elasticsearch query error {e}")
    hits_section = resp.get("hits", {})
    total_hits = 0
    all_hits = []
    aggregations = resp.get("aggregations")
    if do_scroll:
        total_val = hits_section.get("total", 0)
        total_hits = total_val.get("value") if isinstance(total_val, dict) else total_val
        all_hits = hits_section.get("hits", [])
        scroll_id = resp.get("_scroll_id")
        while True:
            scroll_resp = await es.scroll(scroll_id=scroll_id, scroll=req.scroll)
            hits = scroll_resp.get("hits", {}).get("hits", [])
            if not hits:
                break
            all_hits.extend(hits)
        if scroll_id:
            try:
                await es.clear_scroll(scroll_id=scroll_id)
                logger.info("Cleared scroll context %s", scroll_id)
            except Exception as e:
                logger.warning("Failed to clear scroll %s: %s", scroll_id, e)
    else:
        total_val = hits_section.get("total", 0)
        total_hits = total_val.get("value") if isinstance(total_val, dict) else total_val
        all_hits = hits_section.get("hits", [])
    documents = [h.get("_source", {}) for h in all_hits]
    logger.info("Returning %d documents", len(documents))
    return {"total": total_hits, "documents": documents, "aggregations": aggregations}
