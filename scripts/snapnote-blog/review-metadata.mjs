// Verified work metadata supplements missing fields; personal content stays in SnapNote.
const works={
  'snapnote-a0f16e5f-d49f-4ae8-9ef9-9be58b620dba':{
    titles:['Dead Poets Society','死亡诗社'],
    title:'Dead Poets Society',
    release_year:1989,
    source:'https://catalog.afi.com/Film/67915-DEAD-POETS-SOCIETY',
    cover:{
      previous:'https://snapnote-test-1316371247.cos.ap-shanghai.myqcloud.com/SnapNote_Sync/resources/b0ad01c284b7a1f5cc2374e54a7e4da3.png',
      replacement:'/library-covers/dead-poets-society.jpg',
      source:'https://www.impawards.com/1989/dead_poets_society_xxlg.html'
    },
    creators:{
      director:'Peter Weir',
      screenplay:'Tom Schulman',
      producer:'Steven Haft / Paul Junger Witt / Tony Thomas',
      music:'Maurice Jarre',
      cinematographer:'John Seale'
    }
  }
};
const roleOf=key=>key.replace(/^\d+_/,'').replace(/^(writer)$/,'screenplay').replace(/^(composer)$/,'music');

export function supplementReviewMetadata(front){
  const work=works[front.slug];
  if(!work||!work.titles.includes(front.title))return front;
  const creators={...front.creators};
  let order=Math.max(0,...Object.keys(creators).map(key=>Number(key.match(/^\d+/)?.[0]||0)));
  for(const [role,name] of Object.entries(work.creators)){
    const existing=Object.keys(creators).find(key=>roleOf(key)===role);
    if(existing){if(!creators[existing])creators[existing]=name;}
    else creators[`${++order}_${role}`]=name;
  }
  // Replace only the old cover explicitly selected by the user; a future new
  // SnapNote cover still takes precedence. Keep the full theatrical poster ratio.
  const usePoster=work.cover&&[work.cover.previous,work.cover.replacement].includes(front.cover);
  return {...front,title:work.title,release_year:front.release_year??work.release_year,creators,
    ...(usePoster?{cover:work.cover.replacement,cover_source:work.cover.source}:{}),
    metadata_sources:[...new Set([...(front.metadata_sources||[]),work.source])]};
}
