import unittest

from jet.decorators.error import log_exceptions, LoggedException


class TestLogExceptionsDecorator(unittest.TestCase):

    def test_catch_specific_exception_and_raise(self):
        @log_exceptions(AttributeError)
        def raise_attribute_error(x, y):
            raise AttributeError("This is an AttributeError")

        with self.assertRaises(AttributeError) as context:
            raise_attribute_error(10, 20)
        self.assertEqual(str(context.exception), "This is an AttributeError")

    def test_catch_all_exceptions_and_raise(self):
        @log_exceptions()
        def raise_general_exception(x):
            raise RuntimeError("This is a RuntimeError")

        with self.assertRaises(RuntimeError) as context:
            raise_general_exception(50)
        self.assertEqual(str(context.exception), "This is a RuntimeError")

    def test_does_not_catch_unlisted_exception(self):
        @log_exceptions(AttributeError)
        def raise_type_error(x):
            raise TypeError("This is a TypeError")

        with self.assertRaises(TypeError):
            raise_type_error(99)

    def test_generator_catches_exceptions_and_raises(self):
        @log_exceptions(ValueError)
        def gen_func():
            yield 1
            raise ValueError("Error in generator")
            yield 2

        gen = gen_func()
        self.assertEqual(next(gen), 1)
        with self.assertRaises(ValueError):
            next(gen)

    def test_catch_multiple_exceptions(self):
        @log_exceptions(AttributeError, TypeError)
        def raise_multiple_errors(x):
            if x == 1:
                raise AttributeError("Attr Error")
            elif x == 2:
                raise TypeError("Type Error")
            raise ValueError("Other Error")  # Should not be caught

        with self.assertRaises(AttributeError):
            raise_multiple_errors(1)
        with self.assertRaises(TypeError):
            raise_multiple_errors(2)
        with self.assertRaises(ValueError):
            raise_multiple_errors(3)  # Not in decorator, should not be caught


class TestLogExceptionsReturnedArgs(unittest.TestCase):

    def test_catch_specific_exception_and_return_logged_exception(self):
        @log_exceptions(AttributeError, raise_exception=False)
        def raise_attribute_error(x, y, z=5):
            raise AttributeError("This is an AttributeError")

        result = raise_attribute_error(10, 20, z=30)
        self.assertIsInstance(result, LoggedException)
        self.assertEqual(result.args_data, (10, 20))
        self.assertEqual(result.kwargs_data, {'z': 30})
        self.assertEqual(str(result), "This is an AttributeError")

    def test_generator_catches_exceptions_and_returns_logged_exception(self):
        @log_exceptions(ValueError, raise_exception=False)
        def gen_func():
            yield 1
            raise ValueError("Error in generator")
            yield 2

        gen = gen_func()
        self.assertEqual(next(gen), 1)
        result = next(gen)
        self.assertIsInstance(result, LoggedException)
        self.assertEqual(str(result), "Error in generator")

    def test_catch_all_exceptions_and_return_args_kwargs(self):
        @log_exceptions(raise_exception=False)
        def gen_func(x, name="default"):
            raise RuntimeError("This is a RuntimeError")

        result = gen_func(42, name="Alice")
        self.assertIsInstance(result, LoggedException)
        self.assertEqual(result.args_data, (42,))
        self.assertEqual(result.kwargs_data, {'name': "Alice"})

    def test_function_without_exceptions_runs_normally(self):
        @log_exceptions()
        def normal_function(x):
            return x * 2

        self.assertEqual(normal_function(10), 20)

    def test_function_that_returns_none(self):
        @log_exceptions()
        def none_function():
            return None

        self.assertIsNone(none_function())

    def test_generator_without_exceptions_runs_normally(self):
        @log_exceptions()
        def gen_func():
            yield 1
            yield 2
            yield 3

        gen = gen_func()
        self.assertEqual(next(gen), 1)
        self.assertEqual(next(gen), 2)
        self.assertEqual(next(gen), 3)

    def test_empty_function_executes_without_error(self):
        @log_exceptions()
        def empty_func():
            pass

        self.assertIsNone(empty_func())


if __name__ == '__main__':
    unittest.main()
