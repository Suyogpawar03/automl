'use client';
import type { FC, ReactNode } from 'react';
import {
  BarChart2,
  BrainCircuit,
  FileText,
  LayoutGrid,
  ListChecks,
  Settings,
  UploadCloud,
  Wand2,
} from 'lucide-react';
import {
  SidebarProvider,
  Sidebar,
  SidebarHeader,
  SidebarContent,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarFooter,
  SidebarInset,
} from '@/components/ui/sidebar';
import { Logo } from '@/components/icons/logo';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';

export type View = 'upload' | 'analysis' | 'visualize' | 'features' | 'preprocessing' | 'models' | 'report';

interface DashboardLayoutProps {
  children: ReactNode;
  activeView: View;
  onNavigate: (view: View) => void;
  hasDataset: boolean;
}

const navItems = [
  { id: 'upload', label: 'Dataset', icon: UploadCloud },
  { id: 'analysis', label: 'Analysis', icon: BarChart2 },
  { id: 'visualize', label: 'Visualize', icon: LayoutGrid },
  { id: 'features', label: 'Features', icon: ListChecks },
  { id: 'preprocessing', label: 'Preprocess', icon: Wand2 },
  { id: 'models', label: 'Models', icon: BrainCircuit },
  { id: 'report', label: 'Report', icon: FileText },
] as const;


export const DashboardLayout: FC<DashboardLayoutProps> = ({ children, activeView, onNavigate, hasDataset }) => {
  return (
    <SidebarProvider>
      <Sidebar>
        <SidebarHeader>
          <div className="flex items-center gap-2">
            <Logo className="size-8 text-primary" />
            <h1 className="text-xl font-semibold">MLVis</h1>
          </div>
        </SidebarHeader>
        <SidebarContent>
          <SidebarMenu>
            {navItems.map((item) => (
              <SidebarMenuItem key={item.id}>
                <SidebarMenuButton
                  onClick={() => onNavigate(item.id as View)}
                  isActive={activeView === item.id}
                  disabled={item.id !== 'upload' && !hasDataset}
                  tooltip={item.label}
                >
                  <item.icon />
                  <span>{item.label}</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
            ))}
          </SidebarMenu>
        </SidebarContent>
        <SidebarFooter>
          <Separator className="my-2" />
          <SidebarMenu>
            <SidebarMenuItem>
                <SidebarMenuButton tooltip="Settings">
                    <Settings />
                    <span>Settings</span>
                </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarFooter>
      </Sidebar>
      <SidebarInset>
        <main className="flex-1 p-4 md:p-6 lg:p-8">
            {children}
        </main>
      </SidebarInset>
    </SidebarProvider>
  );
};
