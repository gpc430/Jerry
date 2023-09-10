## Analysis Of US Stock Trading Data

git: [source code](https://github.com/gpc430/Analysis-Of-US-Stock-Trading-Data)

Intro：This tool can visually show the highest and lowest prices of stocks in a period of time and the rise and fall range, helping investors to better grasp the stock dynamics. The following information was analyzed to obtain: The tick speed (trades per second) and turnover ( trade amount per second) calculated per second in separate rows;  The cumulative uptick and downtick turnover within that second;  The VWAP ( volume- weighted average price) for that second, which is the cumulative turnover divided by the cumulative volume;  Using the price of the previous trade before time T (referred to as P) , calculate the time points when the price increases or decreases by 0.5%, 1%, 1.5%, or 2% after time T;  Calculate the maximum percentage of price increase and decrease during the observation period N, which starts at time T. A GUI software was created using PyQT5 that incorporates Matplotlib components to display real-time dynamic data of stocks on a per-second basis, as well as visualizations of the key data.
<p align="center">
  <img src="/Jerry/assets/images/WechatIMG27.jpg" width="70%"/>
</p>
