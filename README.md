# Sales Decline Forecasting

Sales Decline Forecasting is a comprehensive system for analyzing and predicting sales dynamics of alcoholic beverages in retail stores. The solution leverages state-of-the-art deep learning techniques, and is designed to work with datasets containing multiple vendors and stores.

## Who Needs This System and Why?

- **Manufacturers and Distributors (e.g., Sazerac):**
  - Forecast demand for their products across different stores and regions.
  - Optimize logistics and inventory.
  - React quickly to sales declines and identify causes (seasonality, competition, assortment changes).
  - Plan marketing campaigns and promotions.

- **Retail Chains and Stores:**
  - Manage inventory to avoid shortages or overstock.
  - Analyze which categories and brands are gaining or losing popularity.
  - Evaluate the effectiveness of working with specific suppliers.
  - Make informed decisions about assortment expansion or reduction.

- **Analysts and Sales Departments:**
  - Receive automated forecasts and sales dynamics reports.
  - Quickly detect anomalies and trends.
  - Assess the impact of external factors (holidays, promotions, weather) on sales.

- **Company Management:**
  - Make strategic decisions based on data-driven forecasts.
  - Evaluate business performance by region, store, and product category.
  - Plan budgets and investments.

## Key Features

- Models are trained on the full dataset, including all vendors and stores, enabling forecasts for any store and brand present in the database.
- Utilizes modern architectures: LSTM with attention, Temporal Fusion Transformer (TFT), and LLMs for generating explanations.
- Handles time series data, store embeddings, feature scaling, and accounts for seasonality and holidays.

## Interactive User Experience

- The system provides an intuitive interface (Streamlit) where users can:
  - Select a store from a dropdown (or enter its ID if it exists in the database).
  - Select a vendor (or view all vendors).
  - Specify the forecast horizon (e.g., 7, 30, or 90 days).
  - Receive a sales forecast, trend graph, and explanations of key factors.
- If a non-existent store is entered, the system prompts the user to select from existing stores or provides an average forecast for a group.

## Example Usage Scenario (for Demo/Presentation)

1. The user selects a store and vendor from the list.
2. The system generates a sales forecast for the chosen period.
3. A graph with predictions and explanations is displayed (e.g., impact of holidays, seasonality, product categories).
4. Users can compare forecasts for different vendors or stores.

## Technical Details

- Modular architecture, easily extensible and scalable.
- All data processing and training steps are logged.
- Uses Airflow for orchestration, MLflow for experiment tracking and Feast for feature management.
- Integration with external services via API is supported.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AAN-innopolis/Sales_Decline_Forecasting.git
   cd Sales_Decline_Forecasting
   ```
2. **Install dependencies:**
   ```bash
   uv sync
   ```
3. **Prepare your data:**
   - Place your dataset in the `data/raw/` directory. Ensure it contains the required columns (see [Data Description](#data-description)).
4. **Run the pipeline:**
   - Use the provided scripts or Airflow DAGs to preprocess data, train models, and generate forecasts.
5. **Launch the interactive interface:**
   - Start the Streamlit app to interact with the system and visualize forecasts.

## Data Description

The dataset should include, at minimum, the following columns:

- `county`: County where the store is located
- `category`: Category code of the liquor ordered
- `category_name`: Category name of the liquor
- `vendor_no`: Vendor number
- `vendor_name`: Vendor name
- `itemno`: Item number
- `im_desc`: Item description
- `pack`: Number of bottles in a case
- `bottle_volume_ml`: Volume of each bottle (ml)
- `state_bottle_cost`: Cost per bottle (wholesale)
- `state_bottle_retail`: Retail price per bottle
- `sale_bottles`: Number of bottles sold
- `sale_dollars`: Total sales amount

## Contributing

Contributions are welcome! Please see [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for guidelines.

## License

This project is licensed under the terms of the [LICENSE](LICENSE).
