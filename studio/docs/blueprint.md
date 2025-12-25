# **App Name**: MLVis

## Core Features:

- Project Management: Create, open, and persist projects, each storing datasets, target features, selected features, models, visualizations, and reports. Includes database functionality.
- Dataset Upload and Validation: Accepts .csv and .xlsx files, displays file information (name, size, row/column count, preview), and automatically validates for missing columns, duplicate rows, empty/constant columns, and invalid data types.
- Automated Data Analysis: Automatically generates column data types, missing value statistics, unique value counts, inconsistent value detection, basic statistics for numerical columns, and frequency distributions for categorical columns.
- Beginner-Friendly Feature Selection: Clearly explains input (X) and target (Y) features. It allows manual and auto-recommended selection, identifies low-dependency features and multicollinearity, detects potential data leakage, and shows simple dependency scores.
- Preprocessing Overview with Toggle Controls: Displays preprocessing steps (missing value handling, categorical encoding, feature scaling, duplicate removal, outlier handling, class imbalance handling) with explanations and impact on training.
- Automated Data Visualization Suite: Automatically generates and displays essential visualizations such as correlation matrix, target variable distribution, feature importance plot, and more, all downloadable as images for project reports. Also has the option to show pair plots, KDE plots, and more. Every visualization must have a clear title, explain what insight it provides, be downloadable as an image, be stored in the project report, and be used to detect outliers, detect inconsistencies, and identify weak or irrelevant features.
- Model Advisor: This tool uses reasoning to automatically determine classification vs. regression and recommend suitable algorithms based on dataset characteristics, explaining the strengths and limitations of each.
- AI Suggestions Section: Generates insights and recommendations on feature engineering, data quality improvements, alternative models, and potential actions for model performance improvements.

## Style Guidelines:

- Primary color: Sea Green (#3CB371), evoking trustworthiness, progress, and technical expertise.
- Background color: Light bluish-green (#E0FFFF), softly muted and calming.
- Accent color: Steel blue (#4682B4), emphasizing call-to-action buttons and interactive elements without distracting from the primary aesthetic.
- Body and headline font: 'Inter', a sans-serif font with a modern and neutral design.
- Code font: 'Source Code Pro' for displaying code snippets.
- Use a set of clear, simple icons from a consistent design language (e.g., Material Design Icons) to represent common actions and data types.
- Clean and modern layout with a dark theme for a professional and user-friendly experience.
- Subtle animations to indicate loading, transitions, and user feedback.