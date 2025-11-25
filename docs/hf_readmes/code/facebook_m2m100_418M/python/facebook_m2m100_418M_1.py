from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

ja_text = "世界各国が水面下で熾烈な情報戦を繰り広げる時代睨み合う2つの国東のオスタニア西のウェスタリス戦争を企てるオスタニア政府要人の動向を探るべくウェスタリスはオペレーションストリックスを発動作戦を担うスゴーデエージェント黄昏百の顔を使い分ける彼の任務は家族を作ること父ロイドフォージャー精神"
chinese_text = "生活就像一盒巧克力。"
hi_text = "जीवन एक चॉकलेट बॉक्स की तरह है।"

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# translate Japanese to French
tokenizer.src_lang = "ja"
target_lang = "en"
encoded_ja = tokenizer(ja_text, return_tensors="pt")
generated_tokens = model.generate(**encoded_ja, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
print(f"Translation {tokenizer.src_lang} -> {target_lang}:")
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
# => "La vie est comme une boîte de chocolat."

# translate Chinese to English
tokenizer.src_lang = "zh"
target_lang = "en"
encoded_zh = tokenizer(chinese_text, return_tensors="pt")
generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
print(f"Translation {tokenizer.src_lang} -> {target_lang}:")
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
# => "Life is like a box of chocolate."

# translate Hindi to French
tokenizer.src_lang = "hi"
target_lang = "fr"
encoded_hi = tokenizer(hi_text, return_tensors="pt")
generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
print(f"Translation {tokenizer.src_lang} -> {target_lang}:")
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
# => "La vie est comme une boîte de chocolat."
