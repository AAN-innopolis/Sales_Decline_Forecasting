# Sales Decline Forecasting

Sales Decline Forecasting is a comprehensive system for analyzing and predicting sales dynamics of alcoholic beverages in retail stores. The solution uses several state-of-the-art deep learning techniques, and is designed to work with datasets containing multiple vendors and stores.

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
- Utilizes modern architectures: LSTM with attention, Temporal Fusion Transformer (TFT), and LLM (GPT-4o mini and Chronos Bolt) for generating explanations.
- Handles time series data, store embeddings, feature scaling, and accounts for seasonality and holidays.

## Interactive User Experience

- The system provides an intuitive interface (Streamlit) where users can:
  - Select a model from a dropdown.
  - Select a store id from a dropdown menu.
  - Receive a sales forecast for the next 30 days.

## Example Usage Scenario (for Demo/Presentation)

1. The user selects a model and a store from the list.
2. The system generates a sales forecast for the chosen store.
3. Users can compare forecasts for different vendors or stores.

## Technical Details

- Modular architecture, easily extensible and scalable.
- All data processing and training steps are logged.
- Uses Airflow for orchestration, Temsorboard for experiment tracking.
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

## About the Dataset

- **Source:** Iowa Department of Revenue, Alcoholic Beverages Division (Commerce)
- **License:** Creative Commons Zero (CC0)
- **Coverage:**
  - **Location:** Iowa, USA
  - **Start Date:** 2012-01-01
  - **Updates:** Data is updated monthly, typically available on the first day of each month.
  - **Rows:** 31.6 million+ (as of May 2025)
  - **Columns:** 24
  - **Each row:** Represents an individual product purchase at the store level (Class E liquor license: grocery stores, liquor stores, convenience stores, etc. — off-premises consumption).
- **Topics:** Sales & Distribution (liquor sales, spirit sales, store sales, liquor licensees)

## Data Description

The dataset should include, at minimum, the following columns (see the [official data portal](https://data.iowa.gov/) for full details):

- `invoice_line_no`: Unique identifier for the individual liquor product in the store order
- `date`: Date of order
- `store`: Unique number assigned to the store
- `name`: Name of the store
- `address`: Address of the store
- `city`: City where the store is located
- `zipcode`: Zip code of the store
- `store_location`: Geographical location (point)
- `county`: County name
- `category`: Category code of the liquor ordered
- `category_name`: Category name of the liquor
- `itemno`: Item number
- `im_desc`: Item description
- `pack`: Number of bottles in a case
- `bottle_volume_ml`: Volume of each bottle (ml)
- `state_bottle_cost`: Cost per bottle (wholesale)
- `sale_bottles`: Number of bottles sold
- `sale_dollars`: Total sales amount
- `sale_liters`: Total volume sold (liters)

> **Note:** For best results, ensure your data is as complete as possible and matches the above schema. The system is designed to handle large-scale, real-world retail sales data.

## Contributing

Contributions are welcome! Please see [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for guidelines.

## License

This project is licensed under the terms of the [LICENSE](LICENSE).
