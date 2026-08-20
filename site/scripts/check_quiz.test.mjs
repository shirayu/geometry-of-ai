import { test } from 'node:test'
import assert from 'node:assert/strict'
import { slugifyHeading } from './check_quiz.mjs'

test('濁点・半濁点付きの見出しは合成済み仮名のスラグになる', () => {
    // slugifyHeading は NFKD 分解後に U+0300-U+036F の結合文字を除去し、続いて NFC で
    // 結合濁点(U+3099)・結合半濁点(U+309A)を基底仮名と再合成する。そのため
    // 「ジ」「ダ」「ポ」等の濁点・半濁点付き仮名は合成済み（見た目そのまま）の形で
    // スラグに残る。quiz.md の S: リンクも合成済み文字列（コピペした見た目そのまま）
    // で書ける。site/.vitepress/config.mts のアンカーslugifyも同じ挙動に揃えてある。
    const slug = slugifyHeading('マージンを決定する「支柱」：サポートベクター')
    assert.equal(slug, 'マージンを決定する「支柱」-サポートベクター')
    assert.notEqual(slug, 'マージンを決定する「支柱」：サポートベクター'.normalize('NFKD'))
})

test('濁点を含まない見出しは通常の文字列と一致する', () => {
    const slug = slugifyHeading('多様体とは何か')
    assert.equal(slug, '多様体とは何か')
})

test('数字始まりの見出しには先頭にアンダースコアが付く', () => {
    const slug = slugifyHeading('2. 数式を恐れるな、しかし数式に溺れるな')
    assert.match(slug, /^_2-/)
})
