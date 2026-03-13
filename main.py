def load_data(filename):
    data = []

    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            row = [float(x) for x in line.split()]
            data.append(row)

    return data

def main():
    filename = input("Enter the dataset filename: ")
    data = load_data(filename)

    num_instances = len(data)
    num_features = len(data[0]) - 1

    print(f"This dataset has {num_instances} instances.")
    print(f"This dataset has {num_features} features (not including the class attribute).")

if __name__ == "__main__":
    main()