"""Expanded V1.5 bench: 100 prompts with automatic evaluation.

Metrics:
  - Exact match (normalized)
  - Key entity match (does the output contain the critical entity?)
  - Domain match (is the output in the right topic area?)
  - BLEU-1/2 against reference
  - Average response length

Each prompt has:
  - prompt: the user query
  - reference: expected correct answer (ground truth)
  - key_entity: the critical word/phrase that MUST appear for "correct"
  - domain: broad category for domain-match scoring
"""
from __future__ import annotations
import json, re, sys, time, argparse, torch
from pathlib import Path
from collections import Counter

sys.path.insert(0, "/workspace/src")


BENCH_PROMPTS = [
    # --- Geography (20) ---
    {"prompt": "What is the capital of Japan?", "reference": "The capital of Japan is Tokyo.", "key_entity": "Tokyo", "domain": "geography"},
    {"prompt": "What is the capital of France?", "reference": "The capital of France is Paris.", "key_entity": "Paris", "domain": "geography"},
    {"prompt": "What is the capital of Brazil?", "reference": "The capital of Brazil is Brasília.", "key_entity": "Bras", "domain": "geography"},
    {"prompt": "What is the capital of Egypt?", "reference": "The capital of Egypt is Cairo.", "key_entity": "Cairo", "domain": "geography"},
    {"prompt": "What is the capital of Australia?", "reference": "The capital of Australia is Canberra.", "key_entity": "Canberra", "domain": "geography"},
    {"prompt": "What is the capital of Canada?", "reference": "The capital of Canada is Ottawa.", "key_entity": "Ottawa", "domain": "geography"},
    {"prompt": "What is the capital of Germany?", "reference": "The capital of Germany is Berlin.", "key_entity": "Berlin", "domain": "geography"},
    {"prompt": "What is the capital of India?", "reference": "The capital of India is New Delhi.", "key_entity": "Delhi", "domain": "geography"},
    {"prompt": "What is the capital of Mexico?", "reference": "The capital of Mexico is Mexico City.", "key_entity": "Mexico City", "domain": "geography"},
    {"prompt": "What is the capital of Peru?", "reference": "The capital of Peru is Lima.", "key_entity": "Lima", "domain": "geography"},
    {"prompt": "What is the largest ocean?", "reference": "The Pacific Ocean is the largest ocean.", "key_entity": "Pacific", "domain": "geography"},
    {"prompt": "What is the longest river?", "reference": "The Nile or Amazon is the longest river.", "key_entity": "Nile|Amazon", "domain": "geography"},
    {"prompt": "What is the tallest mountain?", "reference": "Mount Everest is the tallest mountain.", "key_entity": "Everest", "domain": "geography"},
    {"prompt": "What is the smallest country?", "reference": "Vatican City is the smallest country.", "key_entity": "Vatican", "domain": "geography"},
    {"prompt": "What is the largest desert?", "reference": "The Sahara is the largest hot desert.", "key_entity": "Sahara|Antarctic", "domain": "geography"},
    {"prompt": "What is the capital of South Korea?", "reference": "The capital of South Korea is Seoul.", "key_entity": "Seoul", "domain": "geography"},
    {"prompt": "What is the capital of Thailand?", "reference": "The capital of Thailand is Bangkok.", "key_entity": "Bangkok", "domain": "geography"},
    {"prompt": "What is the capital of Argentina?", "reference": "The capital of Argentina is Buenos Aires.", "key_entity": "Buenos Aires", "domain": "geography"},
    {"prompt": "What is the capital of Turkey?", "reference": "The capital of Turkey is Ankara.", "key_entity": "Ankara", "domain": "geography"},
    {"prompt": "What is the capital of Kenya?", "reference": "The capital of Kenya is Nairobi.", "key_entity": "Nairobi", "domain": "geography"},

    # --- Literature (10) ---
    {"prompt": "Who wrote Hamlet?", "reference": "William Shakespeare wrote Hamlet.", "key_entity": "Shakespeare", "domain": "literature"},
    {"prompt": "Who wrote 1984?", "reference": "George Orwell wrote 1984.", "key_entity": "Orwell", "domain": "literature"},
    {"prompt": "Who wrote Pride and Prejudice?", "reference": "Jane Austen wrote Pride and Prejudice.", "key_entity": "Austen", "domain": "literature"},
    {"prompt": "Who wrote Don Quixote?", "reference": "Miguel de Cervantes wrote Don Quixote.", "key_entity": "Cervantes", "domain": "literature"},
    {"prompt": "Who wrote War and Peace?", "reference": "Leo Tolstoy wrote War and Peace.", "key_entity": "Tolstoy", "domain": "literature"},
    {"prompt": "Who wrote The Great Gatsby?", "reference": "F. Scott Fitzgerald wrote The Great Gatsby.", "key_entity": "Fitzgerald", "domain": "literature"},
    {"prompt": "Who wrote Crime and Punishment?", "reference": "Fyodor Dostoevsky wrote Crime and Punishment.", "key_entity": "Dostoevsky|Dostoyevsky", "domain": "literature"},
    {"prompt": "Who wrote Moby Dick?", "reference": "Herman Melville wrote Moby Dick.", "key_entity": "Melville", "domain": "literature"},
    {"prompt": "Who wrote The Odyssey?", "reference": "Homer wrote The Odyssey.", "key_entity": "Homer", "domain": "literature"},
    {"prompt": "Who wrote Macbeth?", "reference": "William Shakespeare wrote Macbeth.", "key_entity": "Shakespeare", "domain": "literature"},

    # --- Math (15) ---
    {"prompt": "What is 2+2?", "reference": "2+2 equals 4.", "key_entity": "4", "domain": "math"},
    {"prompt": "What is 7*8?", "reference": "7 times 8 equals 56.", "key_entity": "56", "domain": "math"},
    {"prompt": "What is 144 divided by 12?", "reference": "144 divided by 12 equals 12.", "key_entity": "12", "domain": "math"},
    {"prompt": "What is 25% of 200?", "reference": "25% of 200 is 50.", "key_entity": "50", "domain": "math"},
    {"prompt": "What is the square root of 64?", "reference": "The square root of 64 is 8.", "key_entity": "8", "domain": "math"},
    {"prompt": "What is 15+27?", "reference": "15+27 equals 42.", "key_entity": "42", "domain": "math"},
    {"prompt": "What is 100-37?", "reference": "100-37 equals 63.", "key_entity": "63", "domain": "math"},
    {"prompt": "What is 9 squared?", "reference": "9 squared is 81.", "key_entity": "81", "domain": "math"},
    {"prompt": "What is 1000 divided by 8?", "reference": "1000 divided by 8 is 125.", "key_entity": "125", "domain": "math"},
    {"prompt": "What is 3*17?", "reference": "3 times 17 is 51.", "key_entity": "51", "domain": "math"},
    {"prompt": "What is 50% of 360?", "reference": "50% of 360 is 180.", "key_entity": "180", "domain": "math"},
    {"prompt": "What is 12*12?", "reference": "12 times 12 is 144.", "key_entity": "144", "domain": "math"},
    {"prompt": "What is 256 divided by 16?", "reference": "256 divided by 16 is 16.", "key_entity": "16", "domain": "math"},
    {"prompt": "What is 75+125?", "reference": "75+125 equals 200.", "key_entity": "200", "domain": "math"},
    {"prompt": "What is 10% of 5000?", "reference": "10% of 5000 is 500.", "key_entity": "500", "domain": "math"},

    # --- Science (15) ---
    {"prompt": "What is the chemical formula for water?", "reference": "The chemical formula for water is H2O.", "key_entity": "H2O", "domain": "science"},
    {"prompt": "What is the speed of light?", "reference": "The speed of light is approximately 300,000 km/s.", "key_entity": "300", "domain": "science"},
    {"prompt": "What planet is closest to the Sun?", "reference": "Mercury is the closest planet to the Sun.", "key_entity": "Mercury", "domain": "science"},
    {"prompt": "What is the largest planet in our solar system?", "reference": "Jupiter is the largest planet.", "key_entity": "Jupiter", "domain": "science"},
    {"prompt": "What gas do plants absorb?", "reference": "Plants absorb carbon dioxide (CO2).", "key_entity": "carbon dioxide|CO2", "domain": "science"},
    {"prompt": "What is the atomic number of hydrogen?", "reference": "The atomic number of hydrogen is 1.", "key_entity": "1", "domain": "science"},
    {"prompt": "What is DNA an abbreviation for?", "reference": "DNA stands for deoxyribonucleic acid.", "key_entity": "deoxyribonucleic", "domain": "science"},
    {"prompt": "What is the boiling point of water in Celsius?", "reference": "Water boils at 100 degrees Celsius.", "key_entity": "100", "domain": "science"},
    {"prompt": "How many bones are in the human body?", "reference": "There are 206 bones in the adult human body.", "key_entity": "206", "domain": "science"},
    {"prompt": "What is the chemical symbol for gold?", "reference": "The chemical symbol for gold is Au.", "key_entity": "Au", "domain": "science"},
    {"prompt": "Summarize photosynthesis in one sentence.", "reference": "Photosynthesis converts light energy, CO2, and water into glucose and oxygen.", "key_entity": "light|glucose|oxygen", "domain": "science"},
    {"prompt": "What causes tides?", "reference": "Tides are caused by the gravitational pull of the Moon and Sun.", "key_entity": "Moon|gravitational", "domain": "science"},
    {"prompt": "What is the hardest natural substance?", "reference": "Diamond is the hardest natural substance.", "key_entity": "diamond|Diamond", "domain": "science"},
    {"prompt": "What is the most abundant element in the universe?", "reference": "Hydrogen is the most abundant element.", "key_entity": "Hydrogen|hydrogen", "domain": "science"},
    {"prompt": "How many chromosomes do humans have?", "reference": "Humans have 46 chromosomes (23 pairs).", "key_entity": "46", "domain": "science"},

    # --- Translation (10) ---
    {"prompt": "Translate 'good morning' to Japanese.", "reference": "Good morning in Japanese is ohayou gozaimasu.", "key_entity": "ohayou|おはよう", "domain": "translation"},
    {"prompt": "Translate 'thank you' to French.", "reference": "Thank you in French is merci.", "key_entity": "merci|Merci", "domain": "translation"},
    {"prompt": "Translate 'goodbye' to Spanish.", "reference": "Goodbye in Spanish is adiós.", "key_entity": "adiós|adios|Adiós", "domain": "translation"},
    {"prompt": "Translate 'hello' to German.", "reference": "Hello in German is hallo.", "key_entity": "hallo|Hallo|Guten Tag", "domain": "translation"},
    {"prompt": "Translate 'water' to Italian.", "reference": "Water in Italian is acqua.", "key_entity": "acqua", "domain": "translation"},
    {"prompt": "Translate 'cat' to Korean.", "reference": "Cat in Korean is goyangi.", "key_entity": "goyangi|고양이", "domain": "translation"},
    {"prompt": "Translate 'love' to Portuguese.", "reference": "Love in Portuguese is amor.", "key_entity": "amor", "domain": "translation"},
    {"prompt": "Translate 'house' to Chinese.", "reference": "House in Chinese is fangzi.", "key_entity": "房子|fangzi|fáng", "domain": "translation"},
    {"prompt": "Translate 'friend' to Arabic.", "reference": "Friend in Arabic is sadiq.", "key_entity": "sadiq|صديق", "domain": "translation"},
    {"prompt": "Translate 'sun' to Russian.", "reference": "Sun in Russian is solntse.", "key_entity": "solntse|солнце", "domain": "translation"},

    # --- History (10) ---
    {"prompt": "In what year did World War II end?", "reference": "World War II ended in 1945.", "key_entity": "1945", "domain": "history"},
    {"prompt": "Who was the first person to walk on the Moon?", "reference": "Neil Armstrong was the first person on the Moon.", "key_entity": "Armstrong", "domain": "history"},
    {"prompt": "In what year was the Declaration of Independence signed?", "reference": "The Declaration of Independence was signed in 1776.", "key_entity": "1776", "domain": "history"},
    {"prompt": "Who discovered penicillin?", "reference": "Alexander Fleming discovered penicillin.", "key_entity": "Fleming", "domain": "history"},
    {"prompt": "When did the Berlin Wall fall?", "reference": "The Berlin Wall fell in 1989.", "key_entity": "1989", "domain": "history"},
    {"prompt": "Who was the first emperor of China?", "reference": "Qin Shi Huang was the first emperor of China.", "key_entity": "Qin", "domain": "history"},
    {"prompt": "In what year did the Titanic sink?", "reference": "The Titanic sank in 1912.", "key_entity": "1912", "domain": "history"},
    {"prompt": "Who invented the telephone?", "reference": "Alexander Graham Bell invented the telephone.", "key_entity": "Bell", "domain": "history"},
    {"prompt": "When was the French Revolution?", "reference": "The French Revolution began in 1789.", "key_entity": "1789", "domain": "history"},
    {"prompt": "Who painted the Mona Lisa?", "reference": "Leonardo da Vinci painted the Mona Lisa.", "key_entity": "Leonardo|da Vinci|Vinci", "domain": "history"},

    # --- Creative (10) ---
    {"prompt": "Write a haiku about autumn rain.", "reference": "Autumn rain falls soft / Leaves dance in the chilly breeze / Puddles mirror sky", "key_entity": "autumn|rain|fall|leaf|leaves", "domain": "creative"},
    {"prompt": "Write a haiku about the ocean.", "reference": "Endless blue expanse / Waves crash upon sandy shore / Salt air fills my lungs", "key_entity": "ocean|wave|sea|water|shore", "domain": "creative"},
    {"prompt": "Write a haiku about winter snow.", "reference": "Silence fills the air / White blanket covers the earth / Footprints in the snow", "key_entity": "snow|winter|white|cold|ice", "domain": "creative"},
    {"prompt": "Write a haiku about spring flowers.", "reference": "Petals softly bloom / Colors paint the waking earth / Spring has come again", "key_entity": "flower|spring|bloom|petal|blossom", "domain": "creative"},
    {"prompt": "Write a haiku about a mountain.", "reference": "Peaks touch morning clouds / Ancient stone stands weathered tall / Eagles soar above", "key_entity": "mountain|peak|summit|stone|high", "domain": "creative"},
    {"prompt": "Write one sentence describing a sunset.", "reference": "The sky blazed with orange and pink as the sun dipped below the horizon.", "key_entity": "sun|sky|orange|pink|horizon|glow", "domain": "creative"},
    {"prompt": "Write one sentence describing a forest.", "reference": "Tall trees formed a green canopy overhead, with shafts of sunlight piercing through.", "key_entity": "tree|forest|green|leaf|wood", "domain": "creative"},
    {"prompt": "Write one sentence describing a city at night.", "reference": "Neon lights reflected off wet streets as the city hummed with nightlife.", "key_entity": "light|night|city|street|glow|neon", "domain": "creative"},
    {"prompt": "Write a short poem about the moon.", "reference": "Silver moon above, casting gentle light below, the world sleeps in peace.", "key_entity": "moon|silver|night|light|glow", "domain": "creative"},
    {"prompt": "Write a short poem about rain.", "reference": "Drops fall from the sky, tapping gently on the roof, the earth drinks it in.", "key_entity": "rain|drop|water|fall|sky", "domain": "creative"},

    # --- Reasoning/Comparison (10) ---
    {"prompt": "Compare cats and dogs as pets.", "reference": "Cats are more independent while dogs are more social and require more attention.", "key_entity": "cat|dog|independent|social", "domain": "reasoning"},
    {"prompt": "Why is the sky blue?", "reference": "The sky appears blue because of Rayleigh scattering of sunlight by the atmosphere.", "key_entity": "scatter|Rayleigh|atmosphere|wavelength", "domain": "reasoning"},
    {"prompt": "Why do leaves change color in autumn?", "reference": "Leaves change color because chlorophyll breaks down, revealing other pigments.", "key_entity": "chlorophyll|pigment|green", "domain": "reasoning"},
    {"prompt": "How does a refrigerator work?", "reference": "A refrigerator uses a compressor and refrigerant to transfer heat from inside to outside.", "key_entity": "compressor|refrigerant|cool|heat", "domain": "reasoning"},
    {"prompt": "Why does ice float on water?", "reference": "Ice floats because it is less dense than liquid water due to hydrogen bonding.", "key_entity": "dense|density|hydrogen|bond", "domain": "reasoning"},
    {"prompt": "Compare democracy and monarchy.", "reference": "Democracy is governed by elected representatives; monarchy is ruled by a hereditary sovereign.", "key_entity": "elect|vote|king|queen|hereditary|sovereign", "domain": "reasoning"},
    {"prompt": "Why do we dream?", "reference": "Dreams may help process emotions, consolidate memories, and solve problems.", "key_entity": "memory|emotion|brain|sleep|process", "domain": "reasoning"},
    {"prompt": "How do vaccines work?", "reference": "Vaccines train the immune system to recognize and fight specific pathogens.", "key_entity": "immune|antibod|pathogen|virus", "domain": "reasoning"},
    {"prompt": "Why is exercise important?", "reference": "Exercise improves cardiovascular health, strengthens muscles, and boosts mental health.", "key_entity": "health|heart|muscle|mental|cardio", "domain": "reasoning"},
    {"prompt": "Compare Python and JavaScript.", "reference": "Python excels in data science and backend; JavaScript dominates web frontend and full-stack.", "key_entity": "Python|JavaScript|web|data|backend|frontend", "domain": "reasoning"},
]


def bleu_n(reference: str, hypothesis: str, n: int = 1) -> float:
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if len(hyp_tokens) < n:
        return 0.0
    ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1))
    hyp_ngrams = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1))
    matches = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
    total = max(len(hyp_tokens) - n + 1, 1)
    return matches / total


def key_entity_match(output: str, key_entity: str) -> bool:
    for entity in key_entity.split("|"):
        if entity.lower() in output.lower():
            return True
    return False


def domain_keywords():
    return {
        "geography": ["capital", "city", "country", "ocean", "river", "mountain", "desert", "continent"],
        "literature": ["wrote", "author", "novel", "book", "play", "poem", "writer"],
        "math": ["equals", "is", "=", "sum", "product", "divided", "percent", "square"],
        "science": ["element", "planet", "atom", "cell", "energy", "molecule", "chemical", "formula", "species"],
        "translation": ["means", "is", "translat", "language", "word"],
        "history": ["year", "century", "first", "invented", "discovered", "founded", "war", "revolution"],
        "creative": ["poem", "haiku", "rain", "sun", "moon", "snow", "flower", "wave", "tree", "light", "sky"],
        "reasoning": ["because", "due to", "causes", "works by", "compared", "difference", "similar"],
    }


def domain_match(output: str, domain: str) -> bool:
    keywords = domain_keywords().get(domain, [])
    output_lower = output.lower()
    return any(kw in output_lower for kw in keywords)


def run_bench(ckpt_path: str, prompt_tokens: int = 32, max_new: int = 128, output_path: str = "/tmp/bench_v15_100.json"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from router.quantization_aware import fake_quantize_per_token

    EDGE_MODEL = "google/gemma-3-1b-it"
    CLOUD_MODEL = "google/gemma-4-26B-A4B-it"
    K = prompt_tokens
    device = "cuda"

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")

    print(f"[load] edge {EDGE_MODEL}...", flush=True)
    edge_full = AutoModelForCausalLM.from_pretrained(EDGE_MODEL, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16)
    edge_tok = AutoTokenizer.from_pretrained(EDGE_MODEL)
    edge_H = edge_full.config.hidden_size
    for p in edge_full.parameters():
        p.requires_grad = False

    print(f"[load] cloud {CLOUD_MODEL}...", flush=True)
    cloud_full = AutoModelForCausalLM.from_pretrained(CLOUD_MODEL, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16)
    cloud_tok = AutoTokenizer.from_pretrained(CLOUD_MODEL)
    tc = cloud_full.config.text_config if hasattr(cloud_full.config, "text_config") else cloud_full.config
    cloud_H = tc.hidden_size
    for p in cloud_full.parameters():
        p.requires_grad = False
    try:
        cloud_full.config._attn_implementation = "eager"
        if hasattr(cloud_full.config, "text_config"):
            cloud_full.config.text_config._attn_implementation = "eager"
        for m in cloud_full.modules():
            if hasattr(m, "config"):
                m.config._attn_implementation = "eager"
    except Exception:
        pass

    print(f"[load] checkpoint {ckpt_path}...", flush=True)
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    proj = torch.nn.Linear(edge_H, edge_H, bias=False).to(device=device, dtype=torch.bfloat16)
    proj.load_state_dict(ck["proj"])
    mlp = torch.nn.Sequential(
        torch.nn.Linear(edge_H, edge_H * 2, bias=False), torch.nn.GELU(),
        torch.nn.Linear(edge_H * 2, edge_H * 2, bias=False), torch.nn.GELU(),
        torch.nn.Linear(edge_H * 2, K * cloud_H, bias=False),
    ).to(device=device, dtype=torch.bfloat16)
    mlp.load_state_dict(ck["mlp"])
    step = ck.get("step", "?")
    print(f"[load] done step={step} K={K}", flush=True)

    results = []
    for i, entry in enumerate(BENCH_PROMPTS):
        prompt = entry["prompt"]
        t0 = time.perf_counter()

        ids = edge_tok(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
        with torch.no_grad():
            out = edge_full(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True, use_cache=False)
        edge_vec = proj(out.hidden_states[-1][:, -1, :])
        soft = mlp(edge_vec).reshape(1, K, cloud_H)
        soft_q, _ = fake_quantize_per_token(soft, bits=8)

        with torch.no_grad():
            out_ids = cloud_full.generate(
                inputs_embeds=soft_q,
                attention_mask=torch.ones(1, K, dtype=torch.long, device=device),
                max_new_tokens=max_new, do_sample=False,
                pad_token_id=cloud_tok.pad_token_id,
            )
        output = cloud_tok.decode(out_ids[0], skip_special_tokens=True)
        elapsed = time.perf_counter() - t0

        ke_match = key_entity_match(output, entry["key_entity"])
        dm_match = domain_match(output, entry["domain"])
        b1 = bleu_n(entry["reference"], output, 1)
        b2 = bleu_n(entry["reference"], output, 2)

        results.append({
            "prompt": prompt,
            "output": output[:300],
            "reference": entry["reference"],
            "key_entity": entry["key_entity"],
            "domain": entry["domain"],
            "key_entity_match": ke_match,
            "domain_match": dm_match,
            "bleu1": round(b1, 3),
            "bleu2": round(b2, 3),
            "elapsed_ms": round(elapsed * 1000, 1),
        })

        status = "✓" if ke_match else ("~" if dm_match else "✗")
        if (i + 1) % 10 == 0 or i < 3:
            print(f"  [{i+1}/{len(BENCH_PROMPTS)}] {status} {prompt[:40]}... → {output[:80]}...", flush=True)

    # Aggregate
    n = len(results)
    ke_rate = sum(r["key_entity_match"] for r in results) / n
    dm_rate = sum(r["domain_match"] for r in results) / n
    avg_b1 = sum(r["bleu1"] for r in results) / n
    avg_b2 = sum(r["bleu2"] for r in results) / n
    avg_len = sum(len(r["output"]) for r in results) / n

    by_domain = {}
    for r in results:
        d = r["domain"]
        if d not in by_domain:
            by_domain[d] = {"total": 0, "ke_match": 0, "dm_match": 0}
        by_domain[d]["total"] += 1
        by_domain[d]["ke_match"] += int(r["key_entity_match"])
        by_domain[d]["dm_match"] += int(r["domain_match"])
    for d in by_domain:
        t = by_domain[d]["total"]
        by_domain[d]["ke_rate"] = round(by_domain[d]["ke_match"] / t, 3)
        by_domain[d]["dm_rate"] = round(by_domain[d]["dm_match"] / t, 3)

    summary = {
        "checkpoint": ckpt_path,
        "step": step,
        "K": K,
        "n_prompts": n,
        "key_entity_match_rate": round(ke_rate, 3),
        "domain_match_rate": round(dm_rate, 3),
        "avg_bleu1": round(avg_b1, 3),
        "avg_bleu2": round(avg_b2, 3),
        "avg_output_len": round(avg_len, 1),
        "by_domain": by_domain,
    }

    print(f"\n{'='*60}", flush=True)
    print(f"[RESULTS] K={K} step={step}", flush=True)
    print(f"  Key Entity Match: {ke_rate:.1%} ({sum(r['key_entity_match'] for r in results)}/{n})", flush=True)
    print(f"  Domain Match:     {dm_rate:.1%}", flush=True)
    print(f"  BLEU-1:           {avg_b1:.3f}", flush=True)
    print(f"  BLEU-2:           {avg_b2:.3f}", flush=True)
    print(f"  Avg Output Len:   {avg_len:.0f} chars", flush=True)
    print(f"\n  By Domain:", flush=True)
    for d, v in sorted(by_domain.items()):
        print(f"    {d:15s} entity={v['ke_rate']:.0%} domain={v['dm_rate']:.0%} ({v['ke_match']}/{v['total']})", flush=True)
    print(f"{'='*60}", flush=True)

    with open(output_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"[done] -> {output_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=32)
    ap.add_argument("--output", default="/tmp/bench_v15_100.json")
    args = ap.parse_args()
    run_bench(args.checkpoint, prompt_tokens=args.prompt_tokens, output_path=args.output)
