from .detect_repetition import detect_repetition


class TestRepetitionDetector:
    def test_no_repetition_short_text(self):
        current_sequence = "The quick brown fox jumps"
        new_chunk = " over the lazy dog"
        expected = False
        result = detect_repetition(
            current_sequence,
            new_chunk,
            context_size=50,
            repetition_threshold=0.8,
            min_repetition_length=10
        )
        assert result == expected, f"Expected {expected}, got {result}"

    def test_repetition_detected_long_pattern(self):
        current_sequence = "The quick brown fox quick brown fox quick brown"
        new_chunk = " quick brown fox"
        expected = True
        result = detect_repetition(
            current_sequence,
            new_chunk,
            context_size=50,
            repetition_threshold=0.8,
            min_repetition_length=10
        )
        assert result == expected, f"Expected {expected}, got {result}"

    def test_short_chunk_below_min_length(self):
        current_sequence = "The quick brown fox"
        new_chunk = " fox"
        expected = False
        result = detect_repetition(
            current_sequence,
            new_chunk,
            context_size=50,
            repetition_threshold=0.8,
            min_repetition_length=10
        )
        assert result == expected, f"Expected {expected}, got {result}"

    def test_empty_chunk(self):
        current_sequence = "The quick brown fox"
        new_chunk = ""
        expected = False
        result = detect_repetition(
            current_sequence,
            new_chunk,
            context_size=50,
            repetition_threshold=0.8,
            min_repetition_length=10
        )
        assert result == expected, f"Expected {expected}, got {result}"

    def test_small_context_no_repetition(self):
        current_sequence = "Hello world"
        new_chunk = " goodbye world"
        expected = False
        result = detect_repetition(
            current_sequence,
            new_chunk,
            context_size=20,
            repetition_threshold=0.8,
            min_repetition_length=5
        )
        assert result == expected, f"Expected {expected}, got {result}"

    def test_repetition_below_threshold(self):
        current_sequence = "The quick brown fox quick brown"
        new_chunk = " quick brown"
        expected = False
        result = detect_repetition(
            current_sequence,
            new_chunk,
            context_size=50,
            repetition_threshold=0.9,
            min_repetition_length=10
        )
        assert result == expected, f"Expected {expected}, got {result}"

    def test_short_pattern_below_min_length(self):
        current_sequence = "the the the the the"
        new_chunk = " the"
        expected = False
        result = detect_repetition(
            current_sequence,
            new_chunk,
            context_size=50,
            repetition_threshold=0.8,
            min_repetition_length=10
        )
        assert result == expected, f"Expected {expected}, got {result}"

    def test_mixed_repetition_above_threshold(self):
        current_sequence = "Repeat this pattern Repeat this pattern Repeat"
        new_chunk = " this pattern"
        expected = True
        result = detect_repetition(
            current_sequence,
            new_chunk,
            context_size=60,
            repetition_threshold=0.7,
            min_repetition_length=12
        )
        assert result == expected, f"Expected {expected}, got {result}"

    def test_varying_context_size_repetition(self):
        current_sequence = "The cycle repeats The cycle repeats The cycle"
        new_chunk = " repeats"
        expected = True
        result = detect_repetition(
            current_sequence,
            new_chunk,
            context_size=40,
            repetition_threshold=0.75,
            min_repetition_length=10
        )
        assert result == expected, f"Expected {expected}, got {result}"

    def test_low_repetition_threshold_no_repetition(self):
        current_sequence = "The quick brown fox jumps over"
        new_chunk = " the lazy dog"
        expected = False
        result = detect_repetition(
            current_sequence,
            new_chunk,
            context_size=50,
            repetition_threshold=0.5,
            min_repetition_length=10
        )
        assert result == expected, f"Expected {expected}, got {result}"

    def test_high_min_repetition_length_no_repetition(self):
        current_sequence = "Short repeat Short repeat"
        new_chunk = " Short repeat"
        expected = False
        result = detect_repetition(
            current_sequence,
            new_chunk,
            context_size=50,
            repetition_threshold=0.8,
            min_repetition_length=20
        )
        assert result == expected, f"Expected {expected}, got {result}"

    def test_mixed_case_repetition(self):
        current_sequence = "Hello World hello world HELLO WORLD"
        new_chunk = " hello world"
        expected = True
        result = detect_repetition(
            current_sequence,
            new_chunk,
            context_size=50,
            repetition_threshold=0.8,
            min_repetition_length=10
        )
        assert result == expected, f"Expected {expected}, got {result}"

    def test_edge_case_single_word_repetition(self):
        current_sequence = "word word word word"
        new_chunk = " word"
        expected = False
        result = detect_repetition(
            current_sequence,
            new_chunk,
            context_size=50,
            repetition_threshold=0.8,
            min_repetition_length=10
        )
        assert result == expected, f"Expected {expected}, got {result}"

    def test_long_sequence_repetition_large_context(self):
        current_sequence = (
            "The sun sets slowly behind the mountain the sun sets slowly "
            "behind the mountain the sun sets slowly behind the mountain"
        )
        new_chunk = " the sun sets slowly behind"
        expected = True
        result = detect_repetition(
            current_sequence,
            new_chunk,
            context_size=100,
            repetition_threshold=0.8,
            min_repetition_length=20
        )
        assert result == expected, f"Expected {expected}, got {result}"

    def test_long_sequence_partial_repetition(self):
        current_sequence = (
            "In the quiet village the wind whispers softly in the quiet "
            "village the wind whispers softly in the quiet village"
        )
        new_chunk = " the wind whispers softly"
        expected = True
        result = detect_repetition(
            current_sequence,
            new_chunk,
            context_size=80,
            repetition_threshold=0.75,
            min_repetition_length=15
        )
        assert result == expected, f"Expected {expected}, got {result}"

    def test_long_sequence_repetition_low_threshold(self):
        current_sequence = (
            "Over the hills and far away we travel over the hills "
            "and far away we travel over the hills and far away"
        )
        new_chunk = " we travel over the hills"
        expected = True
        result = detect_repetition(
            current_sequence,
            new_chunk,
            context_size=90,
            repetition_threshold=0.6,
            min_repetition_length=15
        )
        assert result == expected, f"Expected {expected}, got {result}"

    def test_long_sequence_no_repetition_despite_length(self):
        current_sequence = (
            "The river flows through the valley and the birds sing "
            "sweetly while the sun shines brightly over the meadows"
        )
        new_chunk = " and the trees sway gently"
        expected = False
        result = detect_repetition(
            current_sequence,
            new_chunk,
            context_size=100,
            repetition_threshold=0.8,
            min_repetition_length=20
        )
        assert result == expected, f"Expected {expected}, got {result}"

    def test_long_sequence_repetition_near_context_boundary(self):
        current_sequence = (
            "Stars twinkle in the night sky stars twinkle in the night "
            "sky stars twinkle in the night sky stars twinkle"
        )
        new_chunk = " in the night sky"
        expected = True
        result = detect_repetition(
            current_sequence,
            new_chunk,
            context_size=60,
            repetition_threshold=0.8,
            min_repetition_length=15
        )
        assert result == expected, f"Expected {expected}, got {result}"

    def test_long_sequence_mixed_pattern_repetition(self):
        current_sequence = (
            "Echoes of the past echo through time echoes of the past "
            "echo through time echoes of the past echo through"
        )
        new_chunk = " time echoes of the past"
        expected = True
        result = detect_repetition(
            current_sequence,
            new_chunk,
            context_size=80,
            repetition_threshold=0.7,
            min_repetition_length=15
        )
        assert result == expected, f"Expected {expected}, got {result}"
