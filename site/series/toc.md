
<script setup>
import { data } from './toc.data.ts'

function isNewPart(index) {
  const page = data[index]
  if (!page.part) return false
  return index === 0 || data[index - 1].part !== page.part
}

// 部ごとに循環する色相（寒色→暖色）。Appendixもグラデーションの最後に含める。
const PART_HUES = [170, 210, 260, 310, 350, 20, 40, 90]
const partOrder = [...new Set(data.map(p => p.part).filter(p => p))]

function partHue(part) {
  if (!part) return null
  const i = partOrder.indexOf(part)
  return PART_HUES[i % PART_HUES.length]
}

function partStyle(part) {
  const hue = partHue(part)
  return hue !== null ? { '--part-hue': hue } : {}
}

function indexInPart(index) {
  let n = 0
  for (let i = 0; i < index; i++) {
    if (data[i].part === data[index].part) n++
  }
  return n
}
</script>

# 全ページ目次

<template v-for="(page, index) in data" :key="page.file">
  <h2 v-if="isNewPart(index)" class="toc-part-title" :style="partStyle(page.part)">{{ page.part }}</h2>
  <div
    :id="page.id"
    class="toc-section"
    :class="{ 'toc-section-hued': partHue(page.part) !== null, 'toc-section-hued-alt': partHue(page.part) !== null && indexInPart(index) % 2 === 1 }"
    :style="partStyle(page.part)"
  >
    <h3 class="toc-page-title">
      <a :href="page.link">{{ page.title }}</a>
    </h3>
    <div class="toc-headings">
      <div
        v-for="h in page.headings"
        :key="h.slug"
        :class="['toc-item', 'toc-h' + h.level]"
      >
        <a :href="page.link + '#' + h.slug">{{ h.text }}</a>
      </div>
      <div v-if="page.quizLink" class="toc-item toc-quiz">
        <a :href="page.quizLink">→ 理解度チェック</a>
      </div>
    </div>
  </div>
</template>

<style scoped>
.toc-part-title {
  margin-top: 2.5rem;
  margin-bottom: 1rem;
}

.toc-part-title:first-child {
  margin-top: 0;
}

.toc-part-title[style*="--part-hue"] {
  color: hsl(var(--part-hue), 45%, 38%);
}

.dark .toc-part-title[style*="--part-hue"] {
  color: hsl(var(--part-hue), 55%, 72%);
}

.toc-section {
  margin-bottom: 1.5rem;
  padding: 1rem 1.25rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background-color: var(--vp-c-bg);
}

.toc-section-hued {
  background-color: hsl(var(--part-hue), 55%, 97%);
  border-color: hsl(var(--part-hue), 40%, 88%);
}

.toc-section-hued-alt {
  background-color: hsl(var(--part-hue), 45%, 92%);
}

.dark .toc-section-hued {
  background-color: hsl(var(--part-hue), 25%, 16%);
  border-color: hsl(var(--part-hue), 25%, 26%);
}

.dark .toc-section-hued-alt {
  background-color: hsl(var(--part-hue), 25%, 21%);
}

.toc-page-title {
  margin-top: 0;
  margin-bottom: 0.5rem;
  padding-left: 0;
  border-left: none;
  font-size: 1.1em;
}

.toc-page-title a {
  text-decoration: none;
}

.toc-page-title a:hover {
  text-decoration: underline;
}

.toc-headings {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
  padding-left: 1.25rem;
}

.toc-item a {
  text-decoration: none;
  font-weight: 400;
}

.toc-item a:hover {
  text-decoration: underline;
}

.toc-h3 {
  padding-left: 1.25rem;
}

.toc-h3 a {
  font-size: 0.875em;
  color: var(--vp-c-text-2);
}

.toc-quiz {
  margin-top: 0.35rem;
}

.toc-quiz a {
  font-size: 0.875em;
  color: var(--vp-c-text-3);
}
</style>
