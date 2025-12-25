import type { FC } from 'react';

interface ViewHeaderProps {
  title: string;
  description: string;
}

export const ViewHeader: FC<ViewHeaderProps> = ({ title, description }) => {
  return (
    <div className="mb-8">
      <h2 className="text-3xl font-bold tracking-tight text-foreground">{title}</h2>
      <p className="mt-2 text-muted-foreground">{description}</p>
    </div>
  );
};
