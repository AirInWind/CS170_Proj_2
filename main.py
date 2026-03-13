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

def leave_one_out_accuracy(data, features):
    correct_count = 0               # to keep track of correctly classified rows
    total_instances = len(data)     # total # of rows in the dataset

    for i in range(total_instances):    # test each row
        best_distance = float("inf")
        nearest_neighbor_label = None

        for j in range(total_instances):    # compare to every other row j
            if i == j:
                continue        # don't compare row to itself
        
            distance = 0.0

            for feature in features:
                difference = data[i][feature] - data[j][feature]
                distance += difference ** 2      # squared Euclidean distance

            if distance < best_distance:
                best_distance = distance
                nearest_neighbor_label = data[j][0]

        if data[i][0] == nearest_neighbor_label:
            correct_count += 1

    return correct_count / total_instances

def forward_selection(data, num_features):
    current_set = []
    best_overall_set = []
    best_overall_accuracy = 0.0

    for level in range(1, num_features + 1):
        feature_to_add = None                   # stores best feature
        best_accuracy_this_level = 0.0          # stores best accuracy

        for feature in range(1, num_features + 1):  # test every feature not in current_set
            if feature in current_set:
                continue

            candidate_set = current_set + [feature]
            accuracy = leave_one_out_accuracy(data, candidate_set)

            print(f"Using feature(s) {candidate_set} accuracy is {accuracy:.3f}")

            if accuracy > best_accuracy_this_level:
                best_accuracy_this_level = accuracy
                feature_to_add = feature
        
        if feature_to_add is not None:              # after testing all candidates, add the best feature
            current_set.append(feature_to_add)

            print(f"Feature set {current_set} was best, accuracy is {best_accuracy_this_level:.3f}\n")

            if best_accuracy_this_level > best_overall_accuracy:
                best_overall_accuracy = best_accuracy_this_level
                best_overall_set = current_set.copy()
    
    return best_overall_set, best_overall_accuracy

def main():
    filename = input("Enter the dataset filename: ")
    data = load_data(filename)

    num_instances = len(data)
    num_features = len(data[0]) - 1

    print(f"This dataset has {num_instances} instances.")
    print(f"This dataset has {num_features} features (not including the class attribute).")

    best_set, best_accuracy = forward_selection(data, num_features)

    print("Finished search.")
    print(f"Best feature subset: {best_set}")
    print(f"Best accuracy: {best_accuracy:.3f}")  

    # accuracy = leave_one_out_accuracy(data, [7, 10, 12])
    # print(f"Accuracy with features [7, 10, 12]: {accuracy:.3f}")

    # filename = input("Enter the dataset filename: ")
    # data = load_data(filename)

    # accuracy = leave_one_out_accuracy(data, [10, 8, 2])
    # print(f"Accuracy with features [7, 10, 12]: {accuracy:.3f}")

if __name__ == "__main__":
    main()