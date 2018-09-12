SELECT 'Query Q07', count(*) as recs from generate_series(10, 1000) a, generate_series(10, 1000) b, generate_series(10, 5000) c where a.value % b.value = 0 and (a.value + b.value ) % c.value = 0;
