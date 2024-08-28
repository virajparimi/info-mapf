---
layout: page
title: Lectures
permalink: /lectures/
---
<ul id="archive">
    {% for lectures in site.data.lectures %}
        <li class="archiveposturl">
            <span>
                <a href="{{ lectures.pptxlink }}">
                    {{ lectures.title }}
                </a>
            </span>
            <br>
            {% if lectures.noteslink %}
                <span class = "postlower">
                    Additional Reading:
                    <a href="{{ lectures.noteslink  }}">
                        Notes
                    </a>
                </span>
                <br>
            {% endif %}

            <span class = "postlower">
                <strong>tl;dr:</strong> {{ lectures.tldr }} 
            </span>
            
            <strong style="font-size:100%; font-family: 'Titillium Web', sans-serif; float:right; padding-right: .5em">
                <a href="{{ lectures.pdflink }}">
                    <i class="fas fa-file-pdf"></i>
                </a>
            </strong> 
        </li>
    {% endfor %}
</ul>
