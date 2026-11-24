from LabOptimizer import LabOptimizer

if __name__ == "__main__":
    optimizer = LabOptimizer(name="MyLabOptimizer", data_file_name="GeneratorMult_data.csv")
    optimizer.train_model()
    print(f"{optimizer.model.feature_importances_}")  # Access feature importances
    optimizer.optimize(budget=350+30, minimuim_allocation=5)  # Adjust budget and minimum allocation as needed