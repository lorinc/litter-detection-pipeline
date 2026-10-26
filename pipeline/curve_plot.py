"""Interactive recall against false positives per photo, from one or more harness results (evaluate.py), as a
Plotly figure: one line per result, hover shows the threshold, recall, false positives and precision at each point,
the operating threshold is a dot, speed per photo is in the legend. All results must carry the same label hash.
usage: curve_plot.py OUT.html RESULT.json[=LABEL] [RESULT.json[=LABEL] ...]     LABEL defaults to the detector name
The page loads plotly.js from its CDN. From a notebook: fig=curve_plot.figure([(label, result_dict), ...]); fig.show().
"""
import json,sys,pathlib
import plotly.graph_objects as go
COLOURS=['#2a78d6','#eb6834','#eda100','#1baf7a','#8d5bd6','#d64c7f']
def figure(results):
    assert len({r['labels_sha256'] for _,r in results})==1,'results are scored on different labels'
    fig=go.Figure()
    for i,(label,r) in enumerate(results):
        c=COLOURS[i%len(COLOURS)]; op=r['operating_point']; s=r['speed']['ms_per_frame_mean']/1000
        name=f'{label}: {s:.0f} s per photo' if s>=10 else f'{label}: {s:.1f} s per photo'
        pts=r['curve']
        fig.add_trace(go.Scatter(x=[p['fp_per_frame'] for p in pts],y=[p['recall'] for p in pts],mode='lines',name=name,
            line=dict(color=c,width=2.5),legendgroup=label,
            customdata=[[p['threshold'],p['precision'] if p['precision'] is not None else float('nan')] for p in pts],
            hovertemplate=f'<b>{label}</b><br>threshold %{{customdata[0]:.3f}}<br>recall %{{y:.1%}}<br>'
                          'false positives per photo %{x:.2f}<br>precision %{customdata[1]:.1%}<extra></extra>'))
        fig.add_trace(go.Scatter(x=[op['fp_per_frame']],y=[op['recall']],mode='markers',showlegend=False,legendgroup=label,
            marker=dict(color=c,size=11,line=dict(color='white',width=2)),
            hovertemplate=f'<b>{label}</b>, operating threshold {r["threshold"]:.2f}<br>recall %{{y:.1%}} '
                          f'({op["found"]} of {op["butts"]} butts)<br>false positives per photo %{{x:.2f}}<extra></extra>'))
    op=results[0][1]['operating_point']
    fig.update_layout(template='plotly_white',hovermode='closest',height=540,margin=dict(l=60,r=20,t=100,b=50),
        title=dict(text=f'Recall against false positives per photo<br><sup>{op["frames"]} {results[0][1]["split"]} photos, '
                        f'{op["butts"]} butts, scored on whole photos, one capture session<br>'
                        'dots at each operating threshold; no intervals on the curves</sup>',font=dict(size=15)),
        xaxis=dict(title='false positives per photo',range=[0,8],rangemode='tozero'),
        yaxis=dict(title='recall',range=[0.6,1.0],tickformat='.0%'),
        legend=dict(orientation='h',yanchor='top',y=-0.15,x=0))
    return fig
if __name__=='__main__':
    out=pathlib.Path(sys.argv[1]); results=[]
    for a in sys.argv[2:]:
        p,_,label=a.partition('='); r=json.load(open(p)); results.append((label or r['detector'],r))
    assert results,'usage: curve_plot.py OUT.html RESULT.json[=LABEL] ...'
    html=figure(results).to_html(include_plotlyjs='cdn',full_html=True,div_id='recall-fp-curve',
                                 config={'responsive':True,'displaylogo':False})
    head='<head><meta charset="utf-8" />'
    assert head in html
    out.write_text(html.replace(head,head+'<meta name="viewport" content="width=device-width, initial-scale=1" />'
                                         '<title>Recall against false positives</title>',1))
    print(out)
