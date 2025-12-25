'use client';
import { useState, type FC, useEffect } from 'react';
import type { Dataset } from '@/lib/types';
import { ViewHeader } from '@/components/views/view-header';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ArrowRight, BrainCircuit, Wand2 } from 'lucide-react';
import { Skeleton } from '@/components/ui/skeleton';
import { View } from '../layout/dashboard-layout';

export interface TrainedModel {
    name: string;
    taskType: string;
}

interface ModelsViewProps {
  dataset: Dataset;
  onNavigate: (view: View) => void;
  onTrainModel: (model: TrainedModel) => void;
}

const recommendations = {
    taskType: 'Classification',
    recommendedAlgorithms: [
        'Logistic Regression',
        'Random Forest',
        'Gradient Boosting (XGBoost)',
        'Support Vector Machine (SVM)'
    ],
    reasoning: "The target variable appears to be binary, making this a classification task. The recommended models offer a range from simple, interpretable models (Logistic Regression) to complex, high-performance ensembles (Random Forest, XGBoost) that handle mixed data types well."
};

export const ModelsView: FC<ModelsViewProps> = ({ dataset, onNavigate, onTrainModel }) => {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => setLoading(false), 1500);
    return () => clearTimeout(timer);
  }, []);

  const handleTrainClick = (algo: string) => {
    onTrainModel({ name: algo, taskType: recommendations.taskType });
  };

  return (
    <div className="space-y-6">
      <ViewHeader
        title="Model Advisor"
        description="AI-powered recommendations for the best models to try on your dataset."
      />
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BrainCircuit className="text-primary" />
            AI Model Recommendations
          </CardTitle>
          <CardDescription>
            Based on your dataset characteristics, here are some suitable algorithms to consider.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="space-y-4">
              <Skeleton className="h-8 w-1/4" />
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-3/4" />
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                <Skeleton className="h-24 w-full" />
                <Skeleton className="h-24 w-full" />
                <Skeleton className="h-24 w-full" />
                <Skeleton className="h-24 w-full" />
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold">Suggested Task Type: <span className="text-primary">{recommendations.taskType}</span></h3>
                <p className="text-muted-foreground mt-1">{recommendations.reasoning}</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {recommendations.recommendedAlgorithms.map(algo => (
                  <div key={algo} className="p-4 rounded-lg border bg-card-foreground/5 flex flex-col justify-between">
                    <h4 className="font-semibold text-lg">{algo}</h4>
                    <div className="flex justify-end mt-2">
                        <Button variant="secondary" size="sm" onClick={() => handleTrainClick(algo)}>
                            <Wand2 className="mr-2 h-4 w-4" />
                            Train Model
                        </Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
      <div className="flex justify-end mt-6">
        <Button size="lg" onClick={() => onNavigate('report')}>
          Next: View Report <ArrowRight className="ml-2 h-4 w-4" />
        </Button>
      </div>
    </div>
  );
};
