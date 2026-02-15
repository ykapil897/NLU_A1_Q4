def load_dataset(filepath, max_samples=1000):
    texts = []
    labels = []

    with open(filepath, "r", encoding="utf-8") as f:
        next(f)  # skip header

        count = 0
        for line in f:

            parts = line.strip().split(",", 1)
            if len(parts) != 2:
                continue

            category = parts[0].strip()
            text = parts[1].strip()

            if category == "sport":
                texts.append(text)
                labels.append(0)
                count += 1

            elif category == "politics":
                texts.append(text)
                labels.append(1)
                count += 1

    return texts, labels
