'use client';
import { useState, type FC } from 'react';
import type { Dataset } from '@/lib/types';
import { ViewHeader } from '@/components/views/view-header';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { ArrowRight, Info } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { View } from '../layout/dashboard-layout';

interface FeatureSelectionViewProps {
  dataset: Dataset;
  onNavigate: (view: View) => void;
}

export const FeatureSelectionView: FC<FeatureSelectionViewProps> = ({ dataset, onNavigate }) => {
  // In a real app, this state would be managed globally or lifted
  const [target, setTarget] = useState<string | null>(dataset.features.target);
  const [inputs, setInputs] = useState<string[]>(dataset.features.inputs);

  const handleInputChange = (column: string, checked: boolean) => {
    setInputs(prev => checked ? [...prev, column] : prev.filter(c => c !== column));
  };
  
  const handleSelectAll = () => {
    setInputs(dataset.features.all.filter(c => c !== target));
  };

  const handleDeselectAll = () => {
    setInputs([]);
  };

  return (
    <div className="space-y-6">
      <ViewHeader
        title="Feature Selection"
        description="Choose the variables you want to predict and the ones to use for prediction."
      />
      
      <Alert className='bg-accent/30 border-accent/50'>
        <Info className="h-4 w-4" />
        <AlertTitle>What are Target and Input Features?</AlertTitle>
        <AlertDescription>
          The <strong>Target (Y)</strong> is the single outcome you want to predict (e.g., 'customer churn'). <strong>Inputs (X)</strong> are all the other features the model will use to make that prediction.
        </AlertDescription>
      </Alert>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>1. Select Target Feature (Y)</CardTitle>
            <CardDescription>Choose the single column your model should learn to predict.</CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[50vh]">
              <RadioGroup value={target ?? undefined} onValueChange={setTarget} className="space-y-2">
                {dataset.features.all.map(col => (
                  <div key={col} className="flex items-center p-3 rounded-md hover:bg-muted">
                    <RadioGroupItem value={col} id={`target-${col}`} />
                    <Label htmlFor={`target-${col}`} className="ml-3 font-normal cursor-pointer flex-1">{col}</Label>
                  </div>
                ))}
              </RadioGroup>
            </ScrollArea>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>2. Select Input Features (X)</CardTitle>
            <CardDescription>Select the columns to use for making predictions.</CardDescription>
            <div className="flex gap-2 pt-2">
                <Button variant="outline" size="sm" onClick={handleSelectAll}>Select All</Button>
                <Button variant="outline" size="sm" onClick={handleDeselectAll}>Deselect All</Button>
            </div>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[50vh]">
              <div className="space-y-2">
              {dataset.features.all.map(col => (
                  <div key={col} className={`flex items-center p-3 rounded-md hover:bg-muted ${col === target ? 'opacity-50' : ''}`}>
                    <Checkbox
                      id={`input-${col}`}
                      checked={inputs.includes(col)}
                      onCheckedChange={(checked) => handleInputChange(col, !!checked)}
                      disabled={col === target}
                    />
                    <Label htmlFor={`input-${col}`} className={`ml-3 font-normal flex-1 ${col === target ? 'cursor-not-allowed' : 'cursor-pointer'}`}>{col}</Label>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>

       <div className="flex justify-end mt-6">
        <Button size="lg" onClick={() => onNavigate('preprocessing')}>
          Next: Preprocessing <ArrowRight className="ml-2 h-4 w-4" />
        </Button>
      </div>
    </div>
  );
};
