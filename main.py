from src.assessment_utilities import load_data, predict_increased_trump_approval

def main():
    print("Start")
    df = load_data("./data/full_dataset.csv")
    results = predict_increased_trump_approval(df)
    print(results)

if __name__ == "__main__":
    main()
