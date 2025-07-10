import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Зареждане на данните
Fi_data = pd.read_csv('Fi.csv')
Fi_data.columns = ['y', 'x', 'Fi']  # Осигуряваме съвместимост на имената на колоните

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
    
    # Ако има само една точка, създаваме константна функция
    if len(group) == 1:
        def constant_func(x_val, y_const=group['y'].iloc[0]):
            return np.full_like(x_val, y_const)
        fi_interpolators[fi] = constant_func
        fi_aggregated_groups[fi] = group
    else:
        # Ако има повече точки, създаваме интерполатор
        f = interp1d(group['x'], group['y'], kind='linear', bounds_error=False, fill_value="extrapolate")
        fi_interpolators[fi] = f
        fi_aggregated_groups[fi] = group

# Подготовка на данните за H
h_values_available = sorted(H_data['H'].unique())
h_to_x = {}
h_to_y_range = {}  # Запазваме диапазона на y за всяко H

for h in h_values_available:
    # Изчисляваме средната x стойност за всяко H
    h_group = H_data[H_data['H'] == h]
    x_val = h_group['x'].mean()
    h_to_x[h] = x_val
    
    # Запазваме диапазона на y за визуализация
    h_to_y_range[h] = (h_group['y'].min(), h_group['y'].max())

# Функция за интерполация на Fi
def interpolate_fi(fi_value):
    # Проверка за точна стойност
    if fi_value in fi_values_available:
        return fi_interpolators[fi_value]
    
    # Намиране на най-близките стойности
    lower_fi = max([f for f in fi_values_available if f <= fi_value], default=None)
    upper_fi = min([f for f in fi_values_available if f >= fi_value], default=None)
    
    if lower_fi is None and upper_fi is None:
        return None
    elif lower_fi is None:
        return fi_interpolators[upper_fi]
    elif upper_fi is None:
        return fi_interpolators[lower_fi]
    
    # Създаване на интерполирана функция
    def interpolated_func(x):
        y_lower = fi_interpolators[lower_fi](x)
        y_upper = fi_interpolators[upper_fi](x)
        
        # Линейна интерполация
        ratio = (fi_value - lower_fi) / (upper_fi - lower_fi)
        return y_lower + ratio * (y_upper - y_lower)
    
    return interpolated_func

# Функция за интерполация на H
def interpolate_h(h_value):
    # Проверка за точна стойност
    if h_value in h_values_available:
        return h_to_x[h_value]
    
    # Намиране на най-близките стойности
    lower_h = max([h for h in h_values_available if h <= h_value], default=None)
    upper_h = min([h for h in h_values_available if h >= h_value], default=None)
    
    if lower_h is None and upper_h is None:
        return None
    elif lower_h is None:
        return h_to_x[upper_h]
    elif upper_h is None:
        return h_to_x[lower_h]
    
    # Линейна интерполация
    ratio = (h_value - lower_h) / (upper_h - lower_h)
    return h_to_x[lower_h] + ratio * (h_to_x[upper_h] - h_to_x[lower_h])

# Функция за намиране на най-близките стойности за визуализация
def find_closest_values(value, available_values):
    if not available_values:
        return None, None
    
    lower = max([v for v in available_values if v <= value], default=None)
    upper = min([v for v in available_values if v >= value], default=None)
    
    return lower, upper

# Streamlit UI
st.title('Номограма за активно напрежение на срязване τb')
st.subheader('BG Pavement Design Guide (2002) - Фиг.9.8')

# Определяне на обхватите за входните полета
min_fi = min(fi_values_available)
max_fi = max(fi_values_available)
min_h = min(h_values_available)
max_h = max(h_values_available)

# Избор на стойности - ръчно въвеждане
col1, col2 = st.columns(2)
with col1:
    fi_value = st.number_input('Стойност за Fi', min_value=min_fi, max_value=max_fi, value=min_fi, step=1.0)
with col2:
    h_value = st.number_input('Стойност за H', min_value=min_h, max_value=max_h, value=min_h, step=10.0)

# Изчисляване на τb с интерполация
f_fi = interpolate_fi(fi_value)
x_h = interpolate_h(h_value)

if f_fi is None or x_h is None:
    st.error("Грешка при интерполация. Моля, проверете входните стойности.")
    st.stop()

y_tau = float(f_fi(x_h))

# Намиране на най-близките стойности за визуализация
closest_fi_lower, closest_fi_upper = find_closest_values(fi_value, fi_values_available)
closest_h_lower, closest_h_upper = find_closest_values(h_value, h_values_available)

# Визуализация
fig, ax = plt.subplots(figsize=(12, 9))

# Намиране на глобални граници
x_min = min(Fi_data['x'].min(), H_data['x'].min()) - 0.001
x_max = max(Fi_data['x'].max(), H_data['x'].max()) + 0.001
y_min = min(Fi_data['y'].min(), H_data['y'].min()) - 0.001
y_max = max(Fi_data['y'].max(), H_data['y'].max()) + 0.001

# Рисуване на всички изолинии за Fi (светли)
for fi_val in fi_values_available:
    group = fi_aggregated_groups[fi_val]
    x_vals = group['x'].values
    y_vals = group['y'].values
    
    if len(group) == 1:
        ax.plot([x_min, x_max], [y_vals[0], y_vals[0]], 'b-', linewidth=0.5, alpha=0.3)
    else:
        x_smooth = np.linspace(x_vals.min(), x_vals.max(), 100)
        f = fi_interpolators[fi_val]
        y_smooth = f(x_smooth)
        ax.plot(x_smooth, y_smooth, 'b-', linewidth=0.5, alpha=0.3)

# Рисуване на най-близките Fi изолинии (удебелени)
for fi_val in [closest_fi_lower, closest_fi_upper]:
    if fi_val is None:
        continue
        
    group = fi_aggregated_groups[fi_val]
    x_vals = group['x'].values
    y_vals = group['y'].values
    
    if len(group) == 1:
        ax.plot([x_min, x_max], [y_vals[0], y_vals[0]], 'b-', linewidth=1.5, alpha=0.7)
    else:
        x_smooth = np.linspace(x_vals.min(), x_vals.max(), 100)
        f = fi_interpolators[fi_val]
        y_smooth = f(x_smooth)
        ax.plot(x_smooth, y_smooth, 'b-', linewidth=1.5, alpha=0.7, 
                label=f'Fi={fi_val} (близка)' if fi_val == closest_fi_lower or fi_val == closest_fi_upper else None)

# Рисуване на всички изолинии за H (светли)
for h_val in h_values_available:
    x_h_line = h_to_x[h_val]
    y_min_h, y_max_h = h_to_y_range[h_val]
    ax.plot([x_h_line, x_h_line], [y_min_h, y_max_h], 'r-', linewidth=0.5, alpha=0.3)

# Рисуване на най-близките H изолинии (удебелени)
for h_val in [closest_h_lower, closest_h_upper]:
    if h_val is None:
        continue
        
    x_h_line = h_to_x[h_val]
    y_min_h, y_max_h = h_to_y_range[h_val]
    ax.plot([x_h_line, x_h_line], [y_min_h, y_max_h], 'r-', linewidth=1.5, alpha=0.7, 
            label=f'H={h_val} (близка)' if h_val == closest_h_lower or h_val == closest_h_upper else None)

# Маркиране на пресечната точка
ax.plot([x_h], [y_tau], 'ro', markersize=8, label=f'τb = {y_tau:.4f}')

# Допълнителни настройки на графиката
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title(f'Номограма Fi-H (Fi={fi_value}, H={h_value})', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Показване на резултата
st.success(f"Изчислена стойност на τb (активно напрежение на срязване): `{y_tau:.4f}`")
st.pyplot(fig)

# Информация за интерполацията
st.divider()
st.subheader('Информация за интерполацията')

if fi_value in fi_values_available:
    st.write(f"- Стойността за **Fi={fi_value}** съвпада с налична изолиния")
else:
    st.write(f"- Стойността за **Fi={fi_value}** се интерполира между Fi={closest_fi_lower} и Fi={closest_fi_upper}")

if h_value in h_values_available:
    st.write(f"- Стойността за **H={h_value}** съвпада с налична изолиния")
else:
    st.write(f"- Стойността за **H={h_value}** се интерполира между H={closest_h_lower} и H={closest_h_upper}")

# Допълнителна информация
st.divider()
st.subheader('Инструкции за използване')
st.write("""
1. Въведете стойност за **Fi** в първото поле (допустими стойности от 5 до 50)
2. Въведете стойност за **H** във второто поле (допустими стойности от 0 до 100)
3. Програмата автоматично ще:
   - Намери най-близките изолинии
   - Извърши линейна интерполация между тях
   - Изчисли τb (y-стойността на пресечната точка)
4. На графиката:
   - Сините линии показват изолиниите за Fi
   - Червените линии показват изолиниите за H
   - Удебелените сини линии маркират най-близките Fi стойности за интерполация
   - Удебелените червени линии маркират най-близките H стойности за интерполация
   - Червената точка показва пресечната точка и стойността на τb
""")

st.subheader('Технически параметри')
st.write(f"Обхват на данните: Fi = {min_fi} до {max_fi}, H = {min_h} до {max_h}")
st.write(f"Брой уникални Fi стойности: {len(fi_values_available)}")
st.write(f"Брой точки за Fi: {len(Fi_data)}")
st.write(f"Брой точки за H: {len(H_data)}")
