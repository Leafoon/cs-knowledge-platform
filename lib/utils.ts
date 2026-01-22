import { type ClassValue, clsx } from "clsx";

export function cn(...inputs: ClassValue[]) {
    return clsx(inputs);
}

export function slugify(text: string): string {
    return text
        .toString()
        .toLowerCase()
        .trim()
        .replace(/\s+/g, '-')
        .replace(/[^\w\-]+/g, '')
        .replace(/\-\-+/g, '-');
}
