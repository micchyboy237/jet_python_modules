import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from jet.libs.txtai.pipeline.tabular import Tabular


class TestTabular(unittest.TestCase):
    def setUp(self):
        self.data = [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25}
        ]
        self.df = pd.DataFrame(self.data)

    def test_init_raises_import_error_when_pandas_not_installed(self):
        with patch("jet.libs.txtai.pipeline.tabular.PANDAS", False):
            with self.assertRaises(ImportError):
                Tabular()

    def test_process_without_idcolumn(self):
        tabular = Tabular(textcolumns=["name", "age"])
        result = tabular.process(self.df)
        expected = [
            (0, "Alice. 30", None),
            (1, "Bob. 25", None)
        ]
        self.assertEqual(result, expected)

    def test_process_with_idcolumn(self):
        tabular = Tabular(idcolumn="id", textcolumns=["name", "age"])
        result = tabular.process(self.df)
        expected = [
            (1, "Alice. 30", None),
            (2, "Bob. 25", None)
        ]
        self.assertEqual(result, expected)

    def test_process_with_content_true(self):
        tabular = Tabular(idcolumn="id", textcolumns=["name"], content=True)
        result = tabular.process(self.df)
        expected = [
            (1, "Alice", None),
            (1, {"id": 1, "name": "Alice", "age": 30}, None),
            (2, "Bob", None),
            (2, {"id": 2, "name": "Bob", "age": 25}, None)
        ]
        self.assertEqual(result, expected)

    def test_concat_creates_correct_text(self):
        tabular = Tabular()
        row = {"name": "Alice", "age": 30}
        result = tabular.concat(row, ["name", "age"])
        self.assertEqual(result, "Alice. 30")

    def test_column_handles_nan_values(self):
        tabular = Tabular()
        result = tabular.column(float("nan"))
        self.assertIsNone(result)

    @patch("pandas.read_csv")
    def test_call_with_csv_file(self, mock_read_csv):
        mock_read_csv.return_value = self.df
        tabular = Tabular(idcolumn="id", textcolumns=["name", "age"])
        result = tabular("data.csv")
        expected = [
            (1, "Alice. 30", None),
            (2, "Bob. 25", None)
        ]
        self.assertEqual(result, expected)
        mock_read_csv.assert_called_once_with("data.csv")


if __name__ == "__main__":
    unittest.main()
