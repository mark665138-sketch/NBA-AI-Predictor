import os, math, datetime as dt
import requests
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title='NBA 智慧預測研究室',page_icon='🏀',layout='wide')
st.title('🏀 NBA 智慧預測研究室')
st.caption('繁體中文｜公開 ESPN 賽程與比分｜可選 The Odds API 即時盤口｜僅供研究，非保證獲利')
ESPN='https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'
TIMEZONE=dt.timezone(dt.timedelta(hours=8))

@st.cache_data(ttl=900,show_spinner=False)
def scoreboard(day):
    r=requests.get(ESPN,params={'dates':day.strftime('%Y%m%d'),'limit':100},timeout=18)
    r.raise_for_status()
    return r.json().get('events',[])

def parse_event(ev):
    comp=(ev.get('competitions') or [{}])[0]
    sides={x.get('homeAway'):x for x in comp.get('competitors',[])}
    if 'home' not in sides or 'away' not in sides:return None
    home,away=sides['home'],sides['away']
    status=ev.get('status',{}).get('type',{}).get('name','')
    utc=pd.to_datetime(ev.get('date'),utc=True)
    return {'id':ev.get('id'),'tipoff':utc.tz_convert('Asia/Taipei'), 'home':home.get('team',{}).get('displayName',''), 'away':away.get('team',{}).get('displayName',''), 'home_id':str(home.get('team',{}).get('id','')), 'away_id':str(away.get('team',{}).get('id','')), 'home_score':float(home.get('score') or 0), 'away_score':float(away.get('score') or 0),'status':status}

@st.cache_data(ttl=3600,show_spinner=False)
def history(end_day,days):
    records=[]
    for i in range(1,days+1):
        d=end_day-dt.timedelta(days=i)
        try:
            for e in scoreboard(d):
                p=parse_event(e)
                if p and p['status']=='STATUS_FINAL':records.append(p)
        except (requests.RequestException,ValueError):continue
    return pd.DataFrame(records)

def team_stats(df,team_id,cutoff=None,window=10):
    if df.empty:return None
    x=df[(df.home_id==team_id)|(df.away_id==team_id)].copy()
    if cutoff is not None:x=x[x.tipoff<cutoff]
    x=x.sort_values('tipoff',ascending=False).head(window)
    if len(x)<3:return None
    ishome=x.home_id==team_id
    pf=np.where(ishome,x.home_score,x.away_score)
    pa=np.where(ishome,x.away_score,x.home_score)
    return {'pf':float(np.mean(pf)),'pa':float(np.mean(pa)),'n':len(x),'sd':float(np.std(pf-pa,ddof=1)) if len(x)>1 else 12.0}

def predict(h,a,home_adjust=2.0,injury_adjust=0.0):
    # Points/game proxy, NOT pace-adjusted offensive rating; shrunk toward league-neutral 113.
    hs=0.5*(h['pf']+a['pa'])+home_adjust/2+injury_adjust/2
    ass=0.5*(a['pf']+h['pa'])-home_adjust/2-injury_adjust/2
    hs=0.8*hs+0.2*113;ass=0.8*ass+0.2*113
    margin=hs-ass
    # Provisional normal margin distribution, sigma=13.5; requires calibration.
    win=0.5*(1+math.erf(margin/(13.5*math.sqrt(2))))
    return hs,ass,win

@st.cache_data(ttl=300,show_spinner=False)
def get_odds(api_key):
    if not api_key:return []
    r=requests.get('https://api.the-odds-api.com/v4/sports/basketball_nba/odds/',params={'apiKey':api_key,'regions':'us','markets':'spreads,totals','oddsFormat':'american'},timeout=20)
    r.raise_for_status()
    return r.json()

def market_for(row,odds):
    # Match by team names; take first available bookmaker as explicit source.
    for e in odds:
        if e.get('home_team')!=row['home'] or e.get('away_team')!=row['away']:continue
        for book in e.get('bookmakers',[]):
            spread=total=None
            for m in book.get('markets',[]):
                if m.get('key')=='spreads':
                    for o in m.get('outcomes',[]):
                        if o.get('name')==row['home']:spread=o.get('point')
                if m.get('key')=='totals':
                    for o in m.get('outcomes',[]):
                        if o.get('name')=='Over':total=o.get('point')
            if spread is not None or total is not None:return spread,total,book.get('title',''),book.get('last_update','')
    return None,None,'',''

with st.sidebar:
    st.header('設定')
    selected=st.date_input('台灣日期',value=dt.datetime.now(TIMEZONE).date())
    lookback=st.slider('歷史觀察天數',30,120,60,10)
    window=st.slider('每隊近期場數',3,15,10)
    home_adv=st.slider('主場分差調整',0.0,5.0,2.0,0.5)
    injury_adj=st.slider('主隊相對傷兵影響（分）',-10.0,10.0,0.0,0.5,help='手動情境假設；不代表已驗證傷兵')
    key=st.text_input('The Odds API Key（選填）',value=os.getenv('ODDS_API_KEY',''),type='password')
    st.info('不填 API Key 時盤口欄位顯示「未取得」，絕不補造。')

try:
    # ESPN uses US calendar dates; fetch adjacent days then filter Taiwan date.
    events=[]
    for offset in (-1,0,1):events.extend(scoreboard(selected+dt.timedelta(days=offset)))
    unique={e.get('id'):e for e in events}
    games=[p for e in unique.values() if (p:=parse_event(e)) and p['tipoff'].date()==selected]
except Exception as exc:
    st.error(f'無法取得 ESPN 賽程：{exc}');st.stop()

st.subheader(f'📅 {selected} 台灣時間賽程')
if not games:st.warning('這一天沒有查到 NBA 賽事，或資料來源暫時未提供。')
try:
    with st.spinner('讀取近期真實賽果（可能需要約 10–30 秒）…'):
        hist=history(selected,lookback)
except Exception as exc:
    st.error(f'歷史資料載入失敗：{exc}');hist=pd.DataFrame()
try:odds=get_odds(key)
except Exception as exc:
    st.warning(f'盤口 API 讀取失敗：{exc}');odds=[]

rows=[]
for g in games:
    h=team_stats(hist,g['home_id'],window=window);a=team_stats(hist,g['away_id'],window=window)
    spread,total,book,updated=market_for(g,odds)
    base={'開賽（台灣）':g['tipoff'].strftime('%m/%d %H:%M'),'客隊':g['away'],'主隊':g['home'],'狀態':g['status'],'主隊勝率':None,'預估比分':'資料不足','模型主隊分差':None,'市場主隊讓分':spread,'讓分差距':None,'市場大小分':total,'大小分差距':None,'信心分數':None,'風險':'高：樣本不足','盤口來源':book or '未取得','盤口更新':updated or '未取得','id':g['id']}
    if h and a:
        hs,ass,p=predict(h,a,home_adv,injury_adj)
        edge=(hs-ass)+float(spread) if spread is not None else None
        te=(hs+ass)-float(total) if total is not None else None
        # Confidence reflects data sufficiency and model disagreement, not empirical success probability.
        confidence=int(max(20,min(80,40+2*min(h['n'],a['n'])+(min(abs(edge),5)*3 if edge is not None else 0)- (10 if injury_adj else 0))))
        risk='高：無市場盤口' if spread is None else ('高：盤口接近或不確定' if abs(edge)<2 else '中：需核對傷兵及盤口')
        base.update({'主隊勝率':round(p*100,1),'預估比分':f'{g["away"]} {ass:.0f} : {g["home"]} {hs:.0f}','模型主隊分差':round(hs-ass,1),'讓分差距':round(edge,1) if edge is not None else None,'大小分差距':round(te,1) if te is not None else None,'信心分數':confidence,'風險':risk})
    rows.append(base)
result=pd.DataFrame(rows)
if not result.empty:
    st.dataframe(result.drop(columns=['id']),use_container_width=True,hide_index=True)
    st.download_button('下載當日分析 CSV',result.to_csv(index=False).encode('utf-8-sig'),file_name=f'nba_predictions_{selected}.csv',mime='text/csv')

st.subheader('🚑 傷兵狀態')
st.warning('目前未串接可靠的逐球員即時傷兵來源。請先核對 NBA 官方 Injury Report，再以側欄「主隊相對傷兵影響」進行情境分析；不應將此假設當成已確認名單。')
st.markdown('[NBA 官方傷兵報告](https://official.nba.com/nba-injury-report-2025-26-season/)（請以官網最新賽季頁面為準）')

st.subheader('📊 歷史滾動回測')
st.caption('每場僅使用該場開賽前的資料，避免直接使用賽後結果；目前是簡易比分模型，不含歷史盤口 ATS 回測。')
if not hist.empty:
    hist=hist.sort_values('tipoff')
    back=[]
    for _,g in hist.iterrows():
        prior=hist[hist.tipoff<g['tipoff']]
        h=team_stats(prior,g['home_id'],window=window);a=team_stats(prior,g['away_id'],window=window)
        if not h or not a:continue
        hs,ass,p=predict(h,a,home_adv,0.0)
        actual_margin=g['home_score']-g['away_score']
        back.append({'日期':g['tipoff'].date(),'對戰':f'{g["away"]} @ {g["home"]}','預估主隊勝率':p,'預估主隊分差':hs-ass,'實際主隊分差':actual_margin,'勝負命中':int((p>=0.5)==(actual_margin>0)),'Brier':(p-int(actual_margin>0))**2,'分差誤差':abs(hs-ass-actual_margin)})
    if back:
        bt=pd.DataFrame(back)
        c1,c2,c3,c4=st.columns(4)
        c1.metric('回測場數',len(bt));c2.metric('勝負準確率',f'{bt["勝負命中"].mean():.1%}');c3.metric('Brier Score',f'{bt["Brier"].mean():.3f}');c4.metric('分差 MAE',f'{bt["分差誤差"].mean():.1f} 分')
        st.dataframe(bt,use_container_width=True,hide_index=True)
        st.download_button('下載回測 CSV',bt.to_csv(index=False).encode('utf-8-sig'),file_name='nba_backtest.csv',mime='text/csv')
    else:st.info('可回測場數不足；請延長歷史觀察天數。')
else:st.info('目前沒有可用歷史比分資料。')

with st.expander('模型定義、限制與風險說明'):
    st.markdown('''**比分預測**：用近期每場得失分交叉估算，再向每隊 113 分的中性先驗收縮 20%；主場優勢預設 2 分。這不是使用每百回合效率的完整模型。\n\n**勝率**：以模型主隊分差及固定 13.5 分標準差的常態分布近似，尚未完成跨賽季校準。\n\n**讓分差距**＝模型主隊預測分差＋市場主隊讓分。例如預測主隊贏 8 分、市場主隊 -5.5，差距 +2.5。**大小分差距**＝模型總分－市場總分。\n\n**信心分數**：示範性資料充分度與盤口差異指標，**不是預測正確率**。\n\n**重要限制**：ESPN 公開端點可能變動或限流；近期樣本可能跨賽季；賽季初小樣本誤差大；盤口僅在提供有效 API Key 時讀取；傷兵不會自動更新；回測不含當時盤口、交易成本或投注報酬；賽前需重新確認。''')
