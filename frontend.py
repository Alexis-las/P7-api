import streamlit as st
import requests

# URL_name = 'http://localhost:8000' # if in local run
URL_name = 'https://alexis-p7-api.onrender.com' # if deployed

def st_step_update(x):
    st.session_state['step'] = x


def st_nb_feat_update():
    if st.session_state['select_nb_feat']:
        st.session_state['nb_feat'] = st.session_state['select_nb_feat']


def main():
    st.title('Streamlit Frontend for P7 API')

    param_init = requests.get(f'{URL_name}/init').json()
    list_id = param_init['client_list']
    list_feat = param_init['feature_list']
    business_threshold = param_init['business_threshold']

    if 'step' not in st.session_state:
        st.session_state['step'] = 0
    if 'nb_feat' not in st.session_state:
        st.session_state['nb_feat'] = len(list_feat)
    if 'feat' not in st.session_state:
        st.session_state['feat'] = list_feat

    # Customer id selection by user
    cust_id = st.sidebar.selectbox('Select client id to focus on', list_id, index=None,
                                   on_change=st_step_update, args=(0,))

    st.session_state['nb_feat'] = st.sidebar.selectbox('Number of features to manage', list(range(len(list_feat) + 1)),
                                                       index=st.session_state['nb_feat'],
                                                       on_change=st_nb_feat_update,
                                                       key='select_nb_feat')

    if cust_id:
        if st.session_state['step'] == 0:
            st.write(f'Client {str(cust_id)} selected:')

        # ------- Client score result
        if st.sidebar.button("Client Score Result", on_click=st_step_update, args=(1,)) or st.session_state[
            'step'] == 1:
            st.header('Client score result', divider='rainbow')
            # Retrieve score information
            st.write('Score result : ')
            score_client = requests.get(f'{URL_name}/client_score/{cust_id}').json()
            # st.write(f"Probability : {score_client['score']}%")
            # st.write(f"business_threshold : {business_threshold}%")

            if score_client['score'] > business_threshold:
                display_colored_box(
                    f"<span style='font-size: 24px; font-weight: bold;'> Rejected <span>  \n\n Probability of having risk: {score_client['score']}% <br> (vs business_threshold: {business_threshold}%)",
                    background_color="red", border_color="darkred",
                    margin=0)
            else:
                display_colored_box(
                    f"<span style='font-size: 24px; font-weight: bold;'> Accepted <span>  \n\n Probability of having risk: {score_client['score']}% <br> (vs business_threshold: {business_threshold}%)",
                    background_color="green", border_color="darkgreen",
                    margin=0)

        # ------- Client score explanation
        if st.sidebar.button("Client Score Explanation", on_click=st_step_update, args=(2,)) or st.session_state[
            'step'] == 2:
            st.header('Client score explanation', divider='rainbow')

            # Retrieve data client explanation
            st.write('Score result explanation : ')
            resp_explain = requests.get(
                f"{URL_name}/client_explain/?id={cust_id}&nb_feat={st.session_state['nb_feat']}").json()
            st.components.v1.html(resp_explain['graph'], width=1000, height=800, scrolling=False)
            st.session_state['feat'] = resp_explain['feat']

        # ------- Features distribution
        if st.sidebar.button("Features Distribution", on_click=st_step_update, args=(3,)) or st.session_state[
            'step'] == 3:
            st.header('Features distribution', divider='rainbow')
            feat_name = st.selectbox('Select feature to focus on', st.session_state['feat'], index=None)

            # Retrieve Features distribution
            if feat_name:
                features_dist = requests.get(f'{URL_name}/features_dist/?id={cust_id}&feat={feat_name}').text
                st.components.v1.html(features_dist, width=1000, height=400, scrolling=True)

        # ------- Client data
        if st.sidebar.button("Client Data", on_click=st_step_update, args=(4,)) or st.session_state['step'] == 4:
            st.header('Client data', divider='rainbow')
            # Retrieve data client information
            st.write('Data client collected : ')
            data_client = requests.get(f'{URL_name}/client_data/{cust_id}').json()
            st.dataframe(data_client['data'])

        # ------- Model explanation
        if st.sidebar.button("Model Explanation", on_click=st_step_update, args=(5,)) or st.session_state['step'] == 5:
            st.header('Model explanation', divider='rainbow')
            # Retrieve data client explanation
            resp_explain = requests.get(
                f"{URL_name}/model_importance/?nb_feat={st.session_state['nb_feat']}").json()
            st.components.v1.html(resp_explain['graph'], width=1000, height=800, scrolling=False)

    else:
        st.header('Welcome')
        st.write('Select a client')


def display_colored_box(text, background_color, border_color, margin):
    styled_text = f'<div style="background-color: {background_color}; border: 2px solid {border_color}; padding: 30px; margin: {margin}px; text-align: center; fontsize: 50">{text}</div>'
    st.markdown(styled_text, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
