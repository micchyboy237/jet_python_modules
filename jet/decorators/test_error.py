import unittest

# Import the decorator
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
        self.assertEqual(result.args_data, (42,))  # Ensure tuple format
        self.assertEqual(result.kwargs_data, {'name': "Alice"})


if __name__ == '__main__':
    unittest.main()
