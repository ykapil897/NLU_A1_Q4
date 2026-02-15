def load_dataset(path, limit=1000):

    texts = []
    labels = []

    with open(path, "r", encoding="utf-8") as file:

        # skip first line (header)
        header = file.readline()

        for line in file:

            if len(texts) >= limit:
                break

            line = line.strip()

            if not line:
                continue

            parts = line.split(",", 1)
            if len(parts) < 2:
                continue

            category = parts[0].strip()
            content = parts[1].strip()

            if category == "sport":
                texts.append(content)
                labels.append(0)

            elif category == "politics":
                texts.append(content)
                labels.append(1)

    return texts, labels
