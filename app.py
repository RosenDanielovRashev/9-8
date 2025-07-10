import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Зареждане на данните
Fi_data = pd.read_csv('Fi.csv')
# Преименуване на колоните за съвместимост
Fi_data.columns = ['y', 'x', 'Fi']

H_data = pd.read_csv('H.csv')

# Премахване на дублиращи се записи
Fi_data = Fi_data.drop_duplicates(subset=['x', 'y', 'Fi'])

# Подготовка на данните за Fi
fi_aggregated_groups = {}
fi_interpolators = {}
fi_values_available = sorted(Fi_data['Fi'].unique())

for fi in fi_values_available:
    group = Fi_data[Fi_data['Fi'] == fi]
    group = group.sort_values(by='x')
    
    fi_aggregated_groups[fi] = group
    
    x = group['x'].values
    y = group['y'].values
    
    if len(x) < 2:
        # Ако има само една точка, създаваме константна функция
        def constant_func(x_val, y_const=y[0]):
            return np.full_like(x_val, y_const)
        fi_interpolators[fi] = constant_func
    else:
        # Ако има повече точки, създаваме интерполатор
        f = interp1d(x, y, kind='linear', bounds_error=False, fill_value="extrapolate")
        fi_interpolators[fi] = f

# Подготовка на данните за H
h_values_available = sorted(H_data['H'].unique())
h_to_x = {}
for h in h_values_available:
    # Вземаме първата x стойност за всяко H
    x_val = H_data[H_data['H'] == h]['x'].iloc[0]
    h_to_x[h] = x_val

# Streamlit UI
st.title('Номограма за активно напрежение на срязване τb')
st.subheader('BG Pavement Design Guide (2002) - Фиг.9.8')

# Избор на стойности
col1, col2 = st.columns(2)
with col1:
    fi_value = st.selectbox('Стойност за Fi', fi_values_available)
with col2:
    h_value = st.selectbox('Стойност за H', h_values_available)

# Изчисляване на τb
x_h = h_to_x[h_value]
f_fi = fi_interpolators[fi_value]
y_tau = float(f_fi(x_h))

# Визуализация
fig, ax = plt.subplots(figsize=(10, 8))

# Намиране на глобални граници
x_min = min(Fi_data['x'].min(), H_data['x'].min()) - 0.001
x_max = max(Fi_data['x'].max(), H_data['x'].max()) + 0.001
y_min = min(Fi_data['y'].min(), H_data['y'].min()) - 0.001
y_max = max(Fi_data['y'].max(), H_data['y'].max()) + 0.001

# Рисуване на всички изолинии за Fi
for fi_val in fi_values_available:
    group = fi_aggregated_groups[fi_val]
    x_vals = group['x'].values
    y_vals = group['y'].values
    
    if len(group) == 1:
        # Рисуване на хоризонтална линия за константни стойности
        if fi_val == fi_value:
            ax.plot([x_min, x_max], [y_vals[0], y_vals[0]], 'b-', linewidth=3, label=f'Fi={fi_val} (избрана)')
        else:
            ax.plot([x_min, x_max], [y_vals[0], y_vals[0]], 'b-', linewidth=1, alpha=0.3)
    else:
        # Рисуване на интерполирани криви
        x_smooth = np.linspace(x_vals.min(), x_vals.max(), 100)
        f = fi_interpolators[fi_val]
        y_smooth = f(x_smooth)
        
        if fi_val == fi_value:
            ax.plot(x_smooth, y_smooth, 'b-', linewidth=3, label=f'Fi={fi_val} (избрана)')
            # Маркиране на оригиналните точки
            ax.scatter(x_vals, y_vals, color='blue', s=50, alpha=0.7)
        else:
            ax.plot(x_smooth, y_smooth, 'b-', linewidth=1, alpha=0.3)

# Рисуване на всички изолинии за H
for h_val in h_values_available:
    x_h_line = h_to_x[h_val]
    if h_val == h_value:
        ax.plot([x_h_line, x_h_line], [y_min, y_max], 'r-', linewidth=3, label=f'H={h_val} (избрана)')
    else:
        ax.plot([x_h_line, x_h_line], [y_min, y_max], 'r-', linewidth=1, alpha=0.3)

# Маркиране на пресечната точка
ax.plot([x_h], [y_tau], 'ro', markersize=8, label=f'τb = {y_tau:.4f}')

# Допълнителни настройки на графиката
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Номограма Fi-H', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Показване на резултата
st.success(f"Изчислена стойност на τb (активно напрежение на срязване): `{y_tau:.4f}`")
st.pyplot(fig)

# Допълнителна информация
st.divider()
st.subheader('Инструкции за използване')
st.write("""
1. Изберете стойност за **Fi** от падащия списък
2. Изберете стойност за **H** от падащия списък
3. Програмата автоматично изчислява τb (y-стойността на пресечната точка)
4. На графиката:
   - Сините линии показват изолиниите за Fi
   - Червените линии показват изолиниите за H
   - Удебелените линии маркират избраните стойности
   - Червената точка показва пресечната точка и стойността на τb
""")

st.subheader('Техническа информация')
st.write(f"Обхват на данните: Fi = {min(fi_values_available)} до {max(fi_values_available)}, H = {min(h_values_available)} до {max(h_values_available)}")
st.write(f"Брой уникални Fi стойности: {len(fi_values_available)}")
st.write(f"Брой точки за Fi: {len(Fi_data)}")
