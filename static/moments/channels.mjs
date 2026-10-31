// Confirmed against the user's SnapNote labels on 2026-10-31.
export const BLOG_TAG = '592f2ffc-d163-4779-aa1b-bdb39d9ebdef';
export const THREADS_TAG = 'bb8f6f57-7e42-44ca-8a28-d8d526d97fbc';
export const REVIEW_TAG = 'ac93333c-e073-4af0-bfd2-19f8b08166cd';
export function channelOf(record) {
  const tags = Array.isArray(record.tagUuids) ? record.tagUuids : [];
  // Review wins over blog; neither channel may leak into threads.
  if (tags.includes(REVIEW_TAG)) return 'review';
  if (tags.includes(BLOG_TAG)) return 'blog';
  return 'threads';
}
