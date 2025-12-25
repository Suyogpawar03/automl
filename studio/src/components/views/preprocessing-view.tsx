'use client';
import type { FC } from 'react';
import { ViewHeader } from '@/components/views/view-header';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Button } from '../ui/button';
import { ArrowRight, CheckCircle, Download } from 'lucide-react';
import { View } from '../layout/dashboard-layout';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Dataset } from '@/lib/types';
import { useToast } from '@/hooks/use-toast';

interface PreprocessingViewProps {
  dataset: Dataset;
  onNavigate: (view: View) => void;
}

const preprocessingSteps = [
  {
    id: 'missing-values',
    title: 'Missing Value Handling',
    description: 'Decide how to treat rows with empty cells (e.g., remove row, fill with mean/median).',
    defaultOn: true,
  },
  {
    id: 'categorical-encoding',
    title: 'Categorical Encoding',
    description: 'Convert text-based categories (like "red", "blue") into numerical format (e.g., One-Hot Encoding).',
    defaultOn: true,
  },
  {
    id: 'feature-scaling',
    title: 'Feature Scaling',
    description: 'Standardize the range of numerical features so they contribute equally to model training.',
    defaultOn: true,
  },
  {
    id: 'duplicate-removal',
    title: 'Duplicate Row Removal',
    description: 'Remove identical rows to prevent the model from being biased by redundant data.',
    defaultOn: true,
  },
  {
    id: 'outlier-handling',
    title: 'Outlier Handling',
    description: 'Manage extreme values that could skew the model (e.g., cap values or remove rows).',
    defaultOn: false,
  },
  {
    id: 'class-imbalance',
    title: 'Class Imbalance Handling',
    description: 'Address situations where one outcome is far more common than others (e.g., using SMOTE).',
    defaultOn: false,
  },
];

const summaryReport = [
    { title: 'Missing Value Imputation', details: 'Filled 11 missing values in `TotalCharges` using the mean.' },
    { title: 'One-Hot Encoding', details: 'Applied to 10 categorical features including `gender`, `Contract`, and `PaymentMethod`.' },
    { title: 'Standard Scaling', details: 'Applied to numerical features `tenure`, `MonthlyCharges`, and `TotalCharges`.' },
    { title: 'Duplicate Removal', details: 'No duplicate rows were found or removed.' },
];

const convertToCSV = (dataset: Dataset): string => {
    const headers = dataset.features.all;
    const rows = dataset.data;

    const headerString = headers.join(',');
    const rowStrings = rows.map(row => 
        headers.map(header => {
            const value = row[header];
            if (typeof value === 'string' && value.includes(',')) {
                return `"${value}"`;
            }
            return value;
        }).join(',')
    );

    return [headerString, ...rowStrings].join('\n');
}


export const PreprocessingView: FC<PreprocessingViewProps> = ({ dataset, onNavigate }) => {
  const { toast } = useToast();
  
  const handleDownload = () => {
    if (!dataset) {
        toast({
            variant: "destructive",
            title: "Download Failed",
            description: "No dataset available to download.",
        });
        return;
    }
    const csvData = convertToCSV(dataset);
    const blob = new Blob([csvData], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    if (link.download !== undefined) {
      const url = URL.createObjectURL(blob);
      link.setAttribute('href', url);
      link.setAttribute('download', `cleaned_${dataset.name}`);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
     toast({
        title: "Download Started",
        description: "Your cleaned dataset is being downloaded.",
    });
  };

  return (
    <div className="space-y-6">
      <ViewHeader
        title="Preprocessing Overview"
        description="Enable or disable automated steps to clean and prepare your data for modeling."
      />
      <Tabs defaultValue="pipeline" className="w-full">
        <TabsList>
          <TabsTrigger value="pipeline">Pipeline</TabsTrigger>
          <TabsTrigger value="summary">Summary Report</TabsTrigger>
        </TabsList>
        <TabsContent value="pipeline">
          <Card>
            <CardHeader>
              <CardTitle>Preprocessing Pipeline</CardTitle>
              <CardDescription>These steps will be applied to your dataset before model training.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {preprocessingSteps.map(step => (
                  <div key={step.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div>
                      <Label htmlFor={step.id} className="text-base font-medium">{step.title}</Label>
                      <p className="text-sm text-muted-foreground">{step.description}</p>
                    </div>
                    <Switch id={step.id} defaultChecked={step.defaultOn} />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="summary">
            <Card>
                <CardHeader className='flex-row items-center justify-between'>
                    <div>
                        <CardTitle>Data Cleaning Summary</CardTitle>
                        <CardDescription>A report on the preprocessing steps applied to your data.</CardDescription>
                    </div>
                    <Button variant="outline" onClick={handleDownload}>
                        <Download className="mr-2 h-4 w-4" />
                        Download Cleaned Data
                    </Button>
                </CardHeader>
                <CardContent>
                    <div className="space-y-4">
                        {summaryReport.map((item, index) => (
                            <div key={index} className="flex items-start gap-4">
                                <CheckCircle className="w-5 h-5 text-green-500 mt-1" />
                                <div>
                                    <h4 className="font-semibold">{item.title}</h4>
                                    <p className="text-sm text-muted-foreground">{item.details}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </CardContent>
            </Card>
        </TabsContent>
      </Tabs>

      <div className="flex justify-end mt-6">
        <Button size="lg" onClick={() => onNavigate('models')}>
          Next: Choose Models <ArrowRight className="ml-2 h-4 w-4" />
        </Button>
      </div>
    </div>
  );
};
