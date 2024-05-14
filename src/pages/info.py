from st_pages import show_pages_from_config, add_page_title
import streamlit as st

# Either this or add_indentation() MUST be called on each page in your
# app to add indendation in the sidebar
add_page_title()
import streamlit as st

def main():
    # Define your array of project members and their tasks
    project_members = {
        "Florian Dima": ["-", "-"],  
        "Spyridon Eftychios Kokotos": ["-", "-"],
        "Nikolaos Balatos": ["-", "-"]
    }

    # Display the array with member names and their tasks
    st.title("Development Team Members and Their Tasks")
    for member, tasks in project_members.items():
        st.header(member)
        for task in tasks:
            st.write("- " + task)

if __name__ == "__main__":
    main()
