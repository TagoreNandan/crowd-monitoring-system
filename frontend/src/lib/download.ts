import { getAuthHeaders } from './api';

export async function downloadFromSrc(src: string, filename: string): Promise<void> {
  let objectUrl: string | null = null;
  try {
    if (/^https?:\/\//i.test(src)) {
      try {
        const response = await fetch(src);
        if (!response.ok) throw new Error(`Download request failed (${response.status})`);
        objectUrl = URL.createObjectURL(await response.blob());
      } catch {
        window.open(src, '_blank', 'noopener,noreferrer');
        return;
      }
    } else {
      const response = await fetch(src, {
        headers: { ...getAuthHeaders() },
      });
      if (!response.ok) throw new Error(`Download request failed (${response.status})`);
      objectUrl = URL.createObjectURL(await response.blob());
    }

    const anchor = document.createElement('a');
    anchor.href = objectUrl;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
  } finally {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
  }
}