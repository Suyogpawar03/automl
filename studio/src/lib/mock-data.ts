import type { Dataset } from './types';

export const MOCK_DATASET: Dataset = {
  name: 'customer_churn.csv',
  size: 145723,
  rows: 7043,
  columnsCount: 21,
  preview: [
    { customerID: '7590-VHVEG', gender: 'Female', SeniorCitizen: 0, Partner: 'Yes', Dependents: 'No', tenure: 1, PhoneService: 'No', MultipleLines: 'No phone service', InternetService: 'DSL' },
    { customerID: '5575-GNVDE', gender: 'Male', SeniorCitizen: 0, Partner: 'No', Dependents: 'No', tenure: 34, PhoneService: 'Yes', MultipleLines: 'No', InternetService: 'DSL' },
    { customerID: '3668-QPYBK', gender: 'Male', SeniorCitizen: 0, Partner: 'No', Dependents: 'No', tenure: 2, PhoneService: 'Yes', MultipleLines: 'No', InternetService: 'DSL' },
    { customerID: '7795-CFOCW', gender: 'Male', SeniorCitizen: 0, Partner: 'No', Dependents: 'No', tenure: 45, PhoneService: 'No', MultipleLines: 'No phone service', InternetService: 'DSL' },
    { customerID: '9237-HQITU', gender: 'Female', SeniorCitizen: 0, Partner: 'No', Dependents: 'No', tenure: 2, PhoneService: 'Yes', MultipleLines: 'No', InternetService: 'Fiber optic' },
  ],
  analysis: {
    columnTypes: {
      customerID: 'string',
      gender: 'categorical',
      SeniorCitizen: 'boolean',
      Partner: 'categorical',
      Dependents: 'categorical',
      tenure: 'numerical',
      PhoneService: 'categorical',
      MultipleLines: 'categorical',
      InternetService: 'categorical',
      OnlineSecurity: 'categorical',
      OnlineBackup: 'categorical',
      DeviceProtection: 'categorical',
      TechSupport: 'categorical',
      StreamingTV: 'categorical',
      StreamingMovies: 'categorical',
      Contract: 'categorical',
      PaperlessBilling: 'categorical',
      PaymentMethod: 'categorical',
      MonthlyCharges: 'numerical',
      TotalCharges: 'numerical',
      Churn: 'categorical',
    },
    dataQuality: [
        { title: 'Missing Values', value: '0.15%', description: 'Found in 11 rows' },
        { title: 'Duplicate Rows', value: '0', description: 'No duplicate entries' },
        { title: 'Empty Columns', value: '0', description: 'All columns have data' },
        { title: 'Inconsistent Values', value: '2', description: 'In `PaymentMethod`' },
    ],
    statistics: {},
  },
  features: {
    all: [
      'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
      'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
      'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
      'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
      'MonthlyCharges', 'TotalCharges', 'Churn'
    ],
    target: 'Churn',
    inputs: [
      'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
      'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
      'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
      'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
      'MonthlyCharges', 'TotalCharges'
    ],
  },
};
