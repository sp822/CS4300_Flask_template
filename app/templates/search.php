<html>
    <head>
        <link rel="stylesheet" href="static/bootstrap.min.css">
        <link rel="stylesheet" href="/static/main.css"/>
        <link rel="stylesheet" href="/static/all.css"/>
    </head>
    <body>
        <div class="topcorner">
            <p>Project Name: {{ name }}</p>
            <p>Student Name: ({{ netid }})</p>
        </div>
        <?php
        if(isset($_Get['submit-btn'])) { ?>
            <form class="form-inline global-search">
                {% if not output.empty %}
                <h1>{{output_message}}</h1>
                {% for y in output %}
                <br>
                <div class= 'title'><h2>{{  y[y.index('Drama Title'): y.index(',')] }}</h2></div>
                <div class = 'Summary'><p>{{  y[y.index('Summary'): y.index(',  Total Similarity')] }}</p></div>
                <div class = 'score'><p>{{  y[y.index('Total Similarity') :] }}</p></div>
                <p></p>
                <hr>
            {% endfor %}
        {% endif %}
        </form>

        <?php
        } else {



        ?>

        <form class="global-search">
            <h1 style="font-size: 55px; font-family:Futura; color: #4285F4">

                <font color=#EA4335>4300</font>
                <font color=#FBBC05>K</font>

                <font color=#34A853>Drama</font>
                <font color=#EA4335>Queen</font>
            </h1>

            <br><br>

            <div class="form-group">
                <label for = "enjoyed">Shows Enjoyed:</label>
                <input id="enjoyed" type="text" name="enjoyed" class="form-control" placeholder="Your Input">
            </div>
            <div class="form-group">
                    <label for = "disliked">Shows Disliked:</label>
                <input id="disliked" type="text" name="disliked" class="form-control" placeholder="Your Input">
            </div>
            <div class="form-group">
                    <label for = "prefered_genres">Preferred Genres:</label>
                <input id="prefered_genres" type="text" name="prefered_genres" class="form-control" placeholder="Your Input">
            </div>
            <div class="form-group">
                    <label for = "prefered_networks">Preferred Network:</label>
                <input id="prefered_networks" type="text" name="prefered_networks" class="form-control" placeholder="Your Input">
            </div>
            <div> <button type="submit" class="btn btn-info" name = 'submit-btn'> Go! </button></div>

        </form>

        <?php
        }
        ?>

    </body>

</html>
