from jet.db import reset_db
from jet.logger import log_sql_results


def test_permanent():
    results = reset_db()
    log_sql_results(results)


def test_temporary():
    results = reset_db(temp=True)
    log_sql_results(results)


if __name__ == '__main__':
    test_permanent()
    test_temporary()
