// Confirmed against the user's SnapNote labels on 2026-10-31.
export const BLOG_TAG = '592f2ffc-d163-4779-aa1b-bdb39d9ebdef';
export const THREADS_TAG = 'bb8f6f57-7e42-44ca-8a28-d8d526d97fbc';
export function channelOf(record) {
  const tags = Array.isArray(record.tagUuids) ? record.tagUuids : [];
  // A blog draft must not accidentally appear in the public timeline.
  if (tags.includes(BLOG_TAG)) return 'blog';
  if (tags.includes(THREADS_TAG)) return 'threads';
  return null;
}
