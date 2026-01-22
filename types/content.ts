export interface Module {
    id: string;
    title: string;
    description: string;
    icon: string;
    color: string;
    chapters?: Chapter[];
    externalLink?: string;
}

export interface Chapter {
    id: string;
    title: string;
    file: string;
    sections?: Section[];
}

export interface Section {
    id: string;
    title: string;
    level: number;
}

export interface TOCItem {
    id: string;
    title: string;
    level: number;
    children?: TOCItem[];
}

export interface ModuleMeta {
    title: string;
    description: string;
    chapters: ChapterMeta[];
}

export interface ChapterMeta {
    id: string;
    title: string;
    file: string;
}
