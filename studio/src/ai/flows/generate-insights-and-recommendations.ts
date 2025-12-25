// 'use server';

/**
 * @fileOverview Generates insights and recommendations on feature engineering,
 * data quality improvements, alternative models, and potential actions for
 * model performance improvements.
 *
 * - generateInsightsAndRecommendations - A function that generates insights and recommendations.
 * - GenerateInsightsAndRecommendationsInput - The input type for the generateInsightsAndRecommendations function.
 * - GenerateInsightsAndRecommendationsOutput - The return type for the generateInsightsAndRecommendations function.
 */

'use server';

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const GenerateInsightsAndRecommendationsInputSchema = z.object({
  datasetDescription: z
    .string()
    .describe('A description of the dataset, including its columns and types.'),
  targetFeature: z.string().describe('The target feature for the model.'),
  selectedFeatures: z.array(z.string()).describe('The features selected for the model.'),
  modelType: z.enum(['classification', 'regression']).describe('The type of model being used.'),
  currentModelPerformance: z
    .string()
    .describe('A description of the current model performance.'),
});

export type GenerateInsightsAndRecommendationsInput = z.infer<
  typeof GenerateInsightsAndRecommendationsInputSchema
>;

const GenerateInsightsAndRecommendationsOutputSchema = z.object({
  insightsAndRecommendations: z.string().describe('Insights and recommendations for improving model performance.'),
});

export type GenerateInsightsAndRecommendationsOutput = z.infer<
  typeof GenerateInsightsAndRecommendationsOutputSchema
>;

export async function generateInsightsAndRecommendations(
  input: GenerateInsightsAndRecommendationsInput
): Promise<GenerateInsightsAndRecommendationsOutput> {
  return generateInsightsAndRecommendationsFlow(input);
}

const prompt = ai.definePrompt({
  name: 'generateInsightsAndRecommendationsPrompt',
  input: {schema: GenerateInsightsAndRecommendationsInputSchema},
  output: {schema: GenerateInsightsAndRecommendationsOutputSchema},
  prompt: `You are an AI expert specializing in machine learning model optimization.

  Based on the following information about the dataset, target feature, selected features, model type, and current model performance, generate insights and recommendations for improving model performance.

  Dataset Description: {{{datasetDescription}}}
  Target Feature: {{{targetFeature}}}
  Selected Features: {{{selectedFeatures}}}
  Model Type: {{{modelType}}}
  Current Model Performance: {{{currentModelPerformance}}}

  Specifically, provide recommendations on:
  - Feature engineering: Suggest new features or transformations of existing features that could improve model performance.
  - Data quality improvements: Identify potential data quality issues and suggest ways to address them.
  - Alternative models: Recommend alternative models that may be better suited for the dataset and task.
  - Potential actions: Suggest specific actions that could be taken to improve model performance.

  Format your response as a concise and actionable list of insights and recommendations.
  `,
});

const generateInsightsAndRecommendationsFlow = ai.defineFlow(
  {
    name: 'generateInsightsAndRecommendationsFlow',
    inputSchema: GenerateInsightsAndRecommendationsInputSchema,
    outputSchema: GenerateInsightsAndRecommendationsOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
