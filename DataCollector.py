import pandas as pd


class DataCollector:
    """Collect and persist experimental data using pandas DataFrames.

    Supports loading previously collected data from a CSV file,
    interactively collecting new observations, and writing data
    back to the configured data source.
    """

    def __init__(self, name='DefaultCollector', data_source=None, columns=None):
        """Initialize the data collector.

        Parameters
        ----------
        name : str, default='DefaultCollector'
            Name assigned to the collector. Also used to generate the
            default CSV filename if data_source is not provided.
        data_source : str or None, default=None
            Path to the CSV file used for reading and writing data.
            Defaults to '<name>_data.csv'.
        columns : list of str or None, default=None
            Column names for the dataset. Defaults to six input
            variables (lab1 through lab6) and one target variable.

        Attributes
        ----------
        name : str
            Collector name.
        data_source : str
            CSV file path.
        columns : list of str
            Dataset column names.
        previously_collected_data : pandas.DataFrame
            Data loaded from the existing CSV file, or an empty
            DataFrame if the file does not exist.
        collected_data : pandas.DataFrame
            Working dataset containing previously loaded and newly
            collected observations.
        """

        if not data_source:
            data_source = f"{name}_data.csv"
        self.name = name
        self.data_source = data_source
        if not columns:
            columns = ['lab1', 'lab2', 'lab3', 'lab4', 'lab5', 'lab6', 'target_value']
        self.columns = columns
        self.previously_collected_data = self.read_data_source()

        self.collected_data = self.previously_collected_data.copy()  # pyright: ignore[reportOptionalMemberAccess]

    def read_data_source(self):
        """Load previously collected data from the configured CSV file.

        Returns
        -------
        pandas.DataFrame
            Data loaded from the CSV file. Returns an empty DataFrame
            with the configured columns if the file does not exist.
        """

        try:
            data = pd.read_csv(self.data_source)
            return data
        except FileNotFoundError:
            print(f"Data source {self.data_source} not found. Initializing with empty data.")
            return pd.DataFrame(columns=self.columns)

    def write_data_source(self, data_overwrite: pd.DataFrame = None):  # pyright: ignore[reportArgumentType]
        """Write collected data to the configured CSV file.

        Parameters
        ----------
        data_overwrite : pandas.DataFrame or None, default=None
            Optional DataFrame to write instead of the collector's
            current dataset. If None, writes self.collected_data.
        """

        if data_overwrite is not None:
            data_to_write = data_overwrite
        else:
            data_to_write = self.collected_data
        data_to_write.to_csv(self.data_source, index=False)
        print(f"Data written to {self.data_source}")

    def collect_data(self):
        """Interactively collect experimental observations.

        Prompts the user to enter comma-separated numeric values
        corresponding to the configured columns. Each valid entry
        is appended to self.collected_data.

        Input collection continues until the user enters 'exit'.

        Notes
        -----
        Each observation must contain exactly as many values as
        there are configured columns. Invalid entries are rejected
        without modifying the dataset.

        Values are converted to floats before being added.
        """

        print("Input data for the following columns:")
        print(f"{self.columns}")
        new_data = ""
        while new_data.lower() != 'exit':
            new_data = input("Enter data as comma-separated values (or type 'exit' to finish): ")
            if new_data.lower() != 'exit':
                try:
                    new_data_list = [float(x.strip()) for x in new_data.split(',')]
                    if len(new_data_list) != len(self.columns):
                        print(f"Please enter exactly {len(self.columns)} values.")
                        continue
                    new_row = pd.DataFrame([new_data_list], columns=self.columns)
                    self.collected_data = pd.concat([self.collected_data, new_row], ignore_index=True)
                except ValueError:
                    print("Invalid input. Please enter numeric values only.")
            else:
                print("Data collection finished.")