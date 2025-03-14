import unittest
from jet.wordnet.lemmatizer import lemmatize_text


class TestLemmatizeText(unittest.TestCase):
    def test_single_word(self):
        result = lemmatize_text("running")
        expected = ["running"]
        self.assertEqual(result, expected)

    def test_sentences(self):
        result = lemmatize_text(
            "The cats were running swiftly in the backyard.")
        expected = [
            "The",
            "cats",
            "were",
            "running",
            "swiftly",
            "in",
            "the",
            "backyard",
            "."
        ]
        self.assertEqual(result, expected)

        result = lemmatize_text(
            "She has better ideas for the upcoming science fair.")
        expected = [
            "She",
            "has",
            "better",
            "ideas",
            "for",
            "the",
            "upcoming",
            "science",
            "fair",
            "."
        ]
        self.assertEqual(result, expected)

        result = lemmatize_text(
            "Experienced programmers are coding sophisticated applications for artificial intelligence.")
        expected = [
            "Experienced",
            "programmers",
            "are",
            "coding",
            "sophisticated",
            "applications",
            "for",
            "artificial",
            "intelligence",
            "."
        ]
        self.assertEqual(result, expected)

        result = lemmatize_text(
            "The children are happily playing in the park despite the rain.")
        expected = [
            "The",
            "children",
            "are",
            "happily",
            "playing",
            "in",
            "the",
            "park",
            "despite",
            "the",
            "rain",
            "."
        ]
        self.assertEqual(result, expected)

    def test_plural_nouns(self):
        result = lemmatize_text("Dogs wolves mice")
        expected = [
            "Dogs",
            "wolves",
            "mice"
        ]
        self.assertEqual(result, expected)

    def test_irregular_verbs(self):
        result = lemmatize_text("went gone going")
        expected = ["went", "gone", "going"]
        self.assertEqual(result, expected)

    def test_unicode_normalization(self):
        # En dash (–) should be converted to "-"
        result = lemmatize_text("Hello – World")
        expected = ["Hello", "-", "World"]
        self.assertEqual(result, expected)

    def test_unicode_characters(self):
        # Should convert accents
        result = lemmatize_text("Café naïve façade résumé")
        expected = ["Cafe", "naive", "facade", "resume"]
        self.assertEqual(result, expected)

    def test_with_software_dev_terms(self):
        sample = "**Job Type:** Full-Time / Project-Based (Remote, PH Applicants Only)\n**Salary:** Competitive (Negotiable Based on Experience)\n---\n### **Job Description:**\nWe seek a skilled developer to build an AI-powered recipe generator, enhancing features beyond DishGen. You will develop the AI backend, intuitive UI/UX, and advanced meal customization options for a premium SaaS platform.\n---\n### **Responsibilities:**\n- Develop an AI-driven recipe generator with personalized meal plans.\n- Build a responsive, mobile-friendly interface.\n- Optimize AI algorithms for speed and accuracy.\n- Implement authentication, payment processing, and database management.\n- Ensure performance, security, and scalability.\n---\n### **Requirements:**\n- Full-stack development (Python, JavaScript, React/Vue, HTML, CSS).\n- Experience with AI/ML (OpenAI API, TensorFlow, NLP).\n- Cloud services (AWS, Firebase) & database management.\n- Strong problem-solving skills & ability to work independently.\n---\n### **Why Join Us?**\n- Work on an exciting AI-driven SaaS project.\n- Competitive salary + performance bonuses.\n- Flexible remote work setup.\n---\n### **How to Apply:**\nSend your resume, portfolio, and a short cover letter. If you've built similar projects, share your work!"
        result = lemmatize_text(sample)
        expected = [
            "React.js",
            "and",
            "JavaScript",
            "are",
            "widely",
            "used",
            "in",
            "modern",
            "web",
            "development",
            "to",
            "build",
            "interactive",
            "user",
            "interfaces",
            "."
        ]

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
