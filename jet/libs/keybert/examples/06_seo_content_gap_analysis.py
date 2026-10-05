from jet.adapters.keybert import KeyBERT

kw_model = KeyBERT()

my_content = """
    Our coffee maker brews fast and tastes great. It has a simple design 
    and is easy to clean. Perfect for home use.
"""

competitor_content = """
    The premium espresso machine features programmable settings, 
    built-in grinder, and milk frother. It offers barista-quality 
    coffee with adjustable temperature control and pressure profiling.
"""

# Extract keywords
my_kws = set(
    [k[0] for k in kw_model.extract_keywords(my_content, stop_words="english")]
)
comp_kws = set(
    [k[0] for k in kw_model.extract_keywords(competitor_content, stop_words="english")]
)

# Find gaps
missing_keywords = comp_kws - my_kws

print("--- My Keywords ---", my_kws)
print("--- Competitor Keywords ---", comp_kws)
print("--- Missing Opportunities ---", missing_keywords)
