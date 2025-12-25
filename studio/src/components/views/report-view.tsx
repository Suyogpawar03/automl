'use client';
import { useState, type FC, useEffect } from 'react';
import type { Dataset } from '@/lib/types';
import { ViewHeader } from '@/components/views/view-header';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Download, Lightbulb, FileText, BarChart, CheckCircle } from 'lucide-react';
import { Skeleton } from '@/components/ui/skeleton';
import { TrainedModel } from './models-view';
import { useToast } from '@/hooks/use-toast';

interface ReportViewProps {
    dataset: Dataset;
    trainedModel: TrainedModel | null;
}

const aiInsights = [
    {
        title: "Feature Engineering",
        suggestion: "Consider creating a 'TenureGroup' feature by binning the 'tenure' column. This could capture non-linear relationships between customer loyalty and churn."
    },
    {
        title: "Data Quality",
        suggestion: "The 'TotalCharges' column has a small percentage of missing values. These appear to correspond with new customers (tenure=0). Consider filling these with 0."
    },
    {
        title: "Alternative Models",
        suggestion: "A LightGBM classifier could offer faster training times and similar performance to XGBoost. It's worth experimenting with if speed is a priority."
    },
    {
        title: "Performance Actions",
        suggestion: "Hyperparameter tuning for the Random Forest model, specifically 'n_estimators' and 'max_depth', is likely to yield significant performance improvements."
    }
];

const mockPerformance = {
    'Classification': [
        { metric: 'Accuracy', score: '91.3%', description: 'Overall correctness of the model.' },
        { metric: 'Precision', score: '88.7%', description: 'Of predicted positives, how many were correct.' },
        { metric: 'Recall', score: '85.4%', description: 'Of actual positives, how many were found.' },
        { metric: 'F1-Score', score: '87.0%', description: 'Harmonic mean of Precision and Recall.' },
    ],
    'Regression': [
        { metric: 'Mean Absolute Error (MAE)', score: '12.51', description: 'Average absolute difference between predicted and actual.' },
        { metric: 'Mean Squared Error (MSE)', score: '231.14', description: 'Average of the squares of the errors.' },
        { metric: 'R-squared (R²)', score: '0.88', description: 'Proportion of variance explained by the model.' },
        { metric: 'Root Mean Squared Error (RMSE)', score: '15.20', description: 'Square root of MSE, in original units.' },
    ]
};

const modelNameToClassName: Record<string, string> = {
    'Logistic Regression': 'LogisticRegression',
    'Random Forest': 'RandomForestClassifier',
    'Gradient Boosting (XGBoost)': 'XGBClassifier',
    'Support Vector Machine (SVM)': 'SVC',
};

export const ReportView: FC<ReportViewProps> = ({ dataset, trainedModel }) => {
    const [loading, setLoading] = useState(true);
    const { toast } = useToast();

    useEffect(() => {
        if (trainedModel) {
            setLoading(true);
            const timer = setTimeout(() => setLoading(false), 1000);
            return () => clearTimeout(timer);
        } else {
            setLoading(false);
        }
    }, [trainedModel]);

    const performanceMetrics = trainedModel 
        ? mockPerformance[trainedModel.taskType as keyof typeof mockPerformance] 
        : mockPerformance['Classification'];
    
    const handleDownloadMetadata = () => {
        if (!trainedModel || !dataset) {
            toast({
                variant: 'destructive',
                title: 'Metadata Not Available',
                description: 'Please train a model first to generate metadata.',
            });
            return;
        }

        const accuracyMetric = performanceMetrics.find(m => m.metric === 'Accuracy');

        const metadata = {
            model_type: modelNameToClassName[trainedModel.name] || trainedModel.name,
            task: trainedModel.taskType.toLowerCase(),
            python_version: "3.10",
            sklearn_version: "1.3.2",
            features: dataset.features.inputs,
            target: dataset.features.target,
            metrics: {
                accuracy: accuracyMetric ? parseFloat(accuracyMetric.score) / 100 : null
            }
        };

        const metadataString = JSON.stringify(metadata, null, 2);
        const blob = new Blob([metadataString], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'model_metadata.json';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);

        toast({
            title: 'Metadata Downloaded',
            description: 'The model_metadata.json file has been saved.',
        });
    };

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-start gap-4">
                <ViewHeader
                    title={trainedModel ? `${trainedModel.name} Performance Report` : "AI-Generated Report"}
                    description="Insights and actionable recommendations to improve your model's performance."
                />
                <div className="flex gap-2 shrink-0">
                    <Button variant="outline" onClick={handleDownloadMetadata}>
                        <Download className="mr-2 h-4 w-4" />
                        Download Metadata
                    </Button>
                    <Button>
                        <FileText className="mr-2 h-4 w-4" />
                        Download Full Report
                    </Button>
                </div>
            </div>

            {trainedModel && (
                 <Card>
                    <CardHeader>
                        <CardTitle className='flex items-center gap-2'>
                            <BarChart className="text-primary" />
                            Model Performance Summary
                        </CardTitle>
                        <CardDescription>Key performance indicators for the trained {trainedModel.name} model.</CardDescription>
                    </CardHeader>
                    <CardContent>
                        {loading ? (
                             <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                <Skeleton className="h-24 w-full" />
                                <Skeleton className="h-24 w-full" />
                                <Skeleton className="h-24 w-full" />
                                <Skeleton className="h-24 w-full" />
                             </div>
                        ) : (
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                {performanceMetrics.map((item) => (
                                    <div key={item.metric} className="p-4 bg-card-foreground/5 rounded-lg">
                                    <p className="text-sm text-muted-foreground">{item.metric}</p>
                                    <p className="text-2xl font-bold">{item.score}</p>
                                    <p className="text-xs text-muted-foreground">{item.description}</p>
                                    </div>
                                ))}
                            </div>
                        )}
                    </CardContent>
                 </Card>
            )}

            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Lightbulb className="text-primary" />
                        Improvement Suggestions
                    </CardTitle>
                    <CardDescription>
                        Our AI has analyzed your project and identified the following areas for potential improvement.
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    {loading && !trainedModel ? (
                        <div className="space-y-6">
                            <div className="space-y-2">
                                <Skeleton className="h-6 w-1/4" />
                                <Skeleton className="h-4 w-full" />
                            </div>
                            <div className="space-y-2">
                                <Skeleton className="h-6 w-1/4" />
                                <Skeleton className="h-4 w-4/5" />
                            </div>
                            <div className="space-y-2">
                                <Skeleton className="h-6 w-1/4" />
                                <Skeleton className="h-4 w-full" />
                            </div>
                        </div>
                    ) : (
                        <div className="space-y-6">
                            {aiInsights.map((insight, index) => (
                                <div key={index} className="flex items-start gap-3">
                                    <CheckCircle className='text-green-500 w-5 h-5 mt-1 shrink-0' />
                                    <div>
                                        <h3 className="font-semibold text-lg">{insight.title}</h3>
                                        <p className="text-muted-foreground mt-1">{insight.suggestion}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    );
};
