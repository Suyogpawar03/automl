'use server';

/**
 * @fileOverview An AI agent that recommends suitable machine learning algorithms based on dataset characteristics.
 *
 * - recommendSuitableAlgorithms - A function that recommends suitable machine learning algorithms.
 * - RecommendSuitableAlgorithmsInput - The input type for the recommendSuitableAlgorithms function.
 * - RecommendSuitableAlgorithmsOutput - The return type for the recommendSuitableAlgorithms function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const RecommendSuitableAlgorithmsInputSchema = z.object({
  datasetDescription: z
    .string()
    .describe('A detailed description of the dataset, including the number of rows, columns, data types, and target variable.'),
  taskType: z
    .enum(['classification', 'regression'])
    .describe('The type of machine learning task (classification or regression).'),
});
export type RecommendSuitableAlgorithmsInput = z.infer<typeof RecommendSuitableAlgorithmsInputSchema>;

const RecommendSuitableAlgorithmsOutputSchema = z.object({
  recommendedAlgorithms: z
    .array(z.string())
    .describe('A list of recommended machine learning algorithms suitable for the dataset.'),
  reasoning: z
    .string()
    .describe('The reasoning behind the algorithm recommendations, considering the dataset characteristics and task type.'),
});
export type RecommendSuitableAlgorithmsOutput = z.infer<typeof RecommendSuitableAlgorithmsOutputSchema>;

export async function recommendSuitableAlgorithms(
  input: RecommendSuitableAlgorithmsInput
): Promise<RecommendSuitableAlgorithmsOutput> {
  return recommendSuitableAlgorithmsFlow(input);
}

const prompt = ai.definePrompt({
  name: 'recommendSuitableAlgorithmsPrompt',
  input: {schema: RecommendSuitableAlgorithmsInputSchema},
  output: {schema: RecommendSuitableAlgorithmsOutputSchema},
  prompt: `You are an expert machine learning model advisor. Based on the dataset description and the type of machine learning task,
you will recommend a list of suitable machine learning algorithms and explain the reasoning behind your recommendations.

Dataset Description: {{{datasetDescription}}}
Task Type: {{{taskType}}}

Recommended Algorithms and Reasoning:`,
});

const recommendSuitableAlgorithmsFlow = ai.defineFlow(
  {
    name: 'recommendSuitableAlgorithmsFlow',
    inputSchema: RecommendSuitableAlgorithmsInputSchema,
    outputSchema: RecommendSuitableAlgorithmsOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
