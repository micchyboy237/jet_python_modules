from jet.db import drop_db
from jet.logger import log_sql_results


def test_sample1():
    results = drop_db("autogenstudio_db_tmp1")
    log_sql_results(results)


if __name__ == '__main__':
    test_sample1()
