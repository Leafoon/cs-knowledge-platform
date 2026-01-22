import { InstructionCycleSimulator } from "@/components/interactive/InstructionCycleSimulator";
import { VonNeumannArchitecture } from "@/components/interactive/VonNeumannArchitecture";
import { ComputerEvolutionTimeline } from "@/components/interactive/ComputerEvolutionTimeline";

interface InteractiveContentProps {
    moduleId: string;
}

export function InteractiveContent({ moduleId }: InteractiveContentProps) {
    if (moduleId !== "computer-organization") {
        return null;
    }

    return (
        <div className="space-y-8">
            {/* Von Neumann Architecture */}
            <div id="interactive-von-neumann">
                <VonNeumannArchitecture />
            </div>

            {/* Instruction Cycle */}
            <div id="interactive-instruction-cycle">
                <InstructionCycleSimulator />
            </div>

            {/* Computer Evolution */}
            <div id="interactive-evolution">
                <ComputerEvolutionTimeline />
            </div>
        </div>
    );
}
