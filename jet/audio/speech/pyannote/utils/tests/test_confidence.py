from jet.audio.speech.pyannote.utils import collect_low_confidence_segments, filter_high_confidence


class TestCollectLowConfidenceSegments:
    def test_collects_segments_below_threshold(self):
        # Given: diarization result with confidence per speaker
        diarization = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "speaker": "S1",
                    "confidence": {"S1": 65.0, "S2": 30.0},
                },
                {
                    "start": 1.0,
                    "end": 2.0,
                    "speaker": "S2",
                    "confidence": {"S1": 80.0, "S2": 90.0},
                },
            ]
        }

        # When: collecting segments below 70
        result = collect_low_confidence_segments(diarization, thresh=70)

        # Then: only first segment should appear
        expected = [
            {
                "start": 0.0,
                "end": 1.0,
                "speaker": "S1",
                "confidence": 65.0,
            }
        ]
        assert result == expected


class TestFilterHighConfidence:
    def test_filters_matches_above_threshold(self):
        # Given: identification results with voiceprint confidence
        ident_result = {
            "voiceprints": [
                {
                    "speaker": "S1",
                    "confidence": {"John": 75.0, "Jane": 20.0},
                },
                {
                    "speaker": "S2",
                    "confidence": {"John": 50.0, "Jane": 40.0},
                },
            ]
        }

        # When: filtering with threshold 60
        result = filter_high_confidence(ident_result, threshold=60)

        # Then: only S1 should match
        expected = [
            {
                "speaker": "S1",
                "match": "John",
                "confidence": 75.0,
            }
        ]
        assert result == expected
