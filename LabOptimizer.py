
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from scipy.optimize import LinearConstraint


class LabOptimizer:
    """Train a regression model and optimize the allocation of points across six labs.

    Uses a Random Forest regression model to approximate the relationship
    between lab allocations and a target value. SciPy's differential
    evolution algorithm searches for allocations that maximize the
    model's predicted target value while satisfying a fixed point budget.
    """

    def __init__(self, name="", data_file_name=""):
        """Initialize the optimizer and its regression model.

        Parameters
        ----------
        name : str, default=""
            Name assigned to the optimizer. Defaults to
            'DefaultLabOptimizer' if no name is provided.
        data_file_name : str, default=""
            Path to the CSV file containing experimental data.
            Defaults to '<name>_data.csv' if no path is provided.

        Attributes
        ----------
        name : str
            Name assigned to the optimizer.
        data_file_name : str
            Path to the experimental dataset.
        model : sklearn.ensemble.RandomForestRegressor
            Regression model used to predict target values from
            lab allocations.
        """
        if not name:
            name = "DefaultLabOptimizer"
            print(f"No name provided. Using default name: {name}")
        self.name = name
        if not data_file_name:
            data_file_name = f"{name}_data.csv"
            print(f"No data file name provided. Using default: {data_file_name}")
        self.data_file_name = data_file_name
        self.model = RandomForestRegressor(
            n_estimators=100
        )

    def load_data(self):
        """Load experimental data from the configured CSV file.

        Returns
        -------
        pandas.DataFrame
            Dataset containing lab allocation features and the
            target_value column.

        Raises
        ------
        FileNotFoundError
            If the configured data file does not exist.
        """
        try:
            data = pd.read_csv(self.data_file_name)
            return data
        except FileNotFoundError:
            print(f"Data file {self.data_file_name} not found. Please ensure the file exists.")
            raise

    def train_model(self):
        """Train the regression model using the experimental dataset.

        Loads the configured dataset, separates the lab allocation
        features from the target variable, and fits the Random Forest
        regression model.

        Notes
        -----
        The dataset must contain a 'target_value' column. All remaining
        columns are treated as model features.

        Updates
        -------
        model : sklearn.ensemble.RandomForestRegressor
            Fits the model using the loaded experimental data.
        """
        data = self.load_data()
        X = data.drop(columns=['target_value'])
        y = data['target_value']
        self.model.fit(X, y)

    def objective(self, allocation):
        """Calculate the negative predicted target value for an allocation.

        Parameters
        ----------
        allocation : array-like of shape (6,)
            Proposed point allocation across the six labs.

        Returns
        -------
        float
            Negative predicted target value. The prediction is negated
            because SciPy's differential evolution algorithm minimizes
            objective functions, while this optimizer seeks to maximize
            the target value.

        Notes
        -----
        The allocation is converted into a pandas DataFrame using the
        feature names recorded during model training.
        """
        prediction = self.model.predict(
            pd.DataFrame([allocation], columns=self.model.feature_names_in_)
        )[0]

        return -prediction

    def optimize(self, budget=320, minimum_allocation=0):
        """Find a point allocation that maximizes the predicted target value.

        Uses differential evolution to search for an optimal distribution
        of points across six labs. Enforces a fixed total point budget
        and a minimum allocation for each lab.

        Parameters
        ----------
        budget : int, default=320
            Total number of points available for allocation. The sum
            of all six lab allocations must equal this value.
        minimum_allocation : int, default=0
            Minimum number of points that must be assigned to each lab.

        Returns
        -------
        scipy.optimize.OptimizeResult
            Optimization result containing the recommended allocation,
            objective function value, convergence information, and
            optimization status.

        Attributes
        ----------
        linear_constraint : scipy.optimize.LinearConstraint
            Constraint requiring the sum of all six allocations to
            equal the specified budget.

        Notes
        -----
        Each allocation is constrained to the interval
        [minimum_allocation, budget] and is required to be an integer.

        The objective function minimizes the negative predicted target
        value, effectively maximizing the model's prediction.

        The regression model must be trained before optimization.
        """
        self.linear_constraint = LinearConstraint(
            np.ones((1, 6)), budget, budget
        )

        result = differential_evolution(
            self.objective,
            bounds=[(minimum_allocation, budget)] * 6,
            constraints=self.linear_constraint,
            integrality=np.ones(6),
            maxiter=200
        )

        print("Recommended allocation:", result.x)
        print("Predicted score:", -result.fun)

        return result
