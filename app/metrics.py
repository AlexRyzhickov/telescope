from prometheus_client import Histogram, Counter

namespace = 'nlp_team'
subsystem = "dual_encoder_embeddings"

REQUEST_TIME = Histogram(
    namespace=namespace,
    subsystem=subsystem,
    name='latency_in_seconds',
    documentation='Response latency histogram',
    labelnames=['status'],
    buckets=[0.01, 0.05, 10],
)
    

REQUEST_COUNT = Counter(
    namespace=namespace,
    subsystem=subsystem,
    name='requests_total',
    documentation='Total HTTP requests',
    labelnames=['status'],
)