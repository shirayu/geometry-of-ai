import { test } from 'node:test'
import assert from 'node:assert/strict'
import { slugifyHeading } from './check_quiz.mjs'

test('濁点・半濁点付きの見出しはNFKD分解された形のスラグになる', () => {
    // check_quiz.mjs の slugifyHeading は NFKD 正規化後に U+0300-U+036F の結合文字だけを
    // 除去する。日本語の結合濁点(U+3099)・結合半濁点(U+309A)はこの範囲外のため、
    // 「ジ」「ダ」「ポ」等の濁点・半濁点付き仮名を含む見出しは、スラグ内で
    // 基底文字+結合文字に分解されたまま残る。quiz.md の S: リンクはこの分解形式で
    // 書く必要があり、通常の合成済み文字列（コピペした見た目そのまま）で書くと
    // headingSlugs() の集合と一致せずチェックに失敗する。
    const slug = slugifyHeading('マージンを決定する「支柱」：サポートベクター')
    assert.equal(slug, 'マージンを決定する「支柱」-サポートベクター'.normalize('NFKD'))
    assert.notEqual(slug, 'マージンを決定する「支柱」-サポートベクター')
})

test('濁点を含まない見出しは通常の文字列と一致する', () => {
    const slug = slugifyHeading('多様体とは何か')
    assert.equal(slug, '多様体とは何か')
})

test('数字始まりの見出しには先頭にアンダースコアが付く', () => {
    const slug = slugifyHeading('2. 数式を恐れるな、しかし数式に溺れるな')
    assert.match(slug, /^_2-/)
})
