# Canonical single-request stock control

This candidate performs no optimization.  It invokes the same stock
DeepGEMM paged-MQA path as the reference with the corrected canonical ABI:
one request, `m` query tokens, one page-table row.
